MODULE HandFollow
    !***********************************************************************
    ! YuMi follows the operator's wrists sent by the laptop (run.py).
    !
    ! Load this SAME module into both T_ROB_L and T_ROB_R. Each task opens its
    ! own TCP server: T_ROB_L on PORT_L, T_ROB_R on PORT_R.
    ! Requires RobotWare option 616-1 PC Interface (socket messaging), plus
    ! the ABB SmartGripper add-in (g_* interface) in both motion tasks.
    ! Commission/calibrate both grippers before running this routine. There
    ! is deliberately no automatic initialization, homing or calibration.
    ! Uses the plane work object already declared in each task:
    !   T_ROB_L -> WOBJ_NAME_L, T_ROB_R -> WOBJ_NAME_R (copied into wFollow).
    !
    ! Protocol (one request/reply per motion step, so data never piles up):
    !   robot  -> laptop : "R,G\0A" (opt in to the gripper field)
    !   laptop -> robot  : "<valid>,<x_mm>,<y_mm>,<grip>\0A"
    !   e.g. "1,12.5,-40.0,1". Three-field replies also work (grip HOLD).
    !   valid = 1 : move to (x, y) in wobjPlaneXY, relative to the center point
    !   valid = 0 : hand lost or robot disarmed -> no new motion/grip command
    !   grip = 1 : open; 0 : close inward; -1 : no new gripper command
    ! Install the matching new laptop code: old code rejects R,G with HOLD.
    ! HOLD never opens a held object, and does not cancel an accepted MoveL
    ! or an already-running gripper operation. No gripper feedback is sent.
    !
    ! Start: jog both arms to the point that should be the center (0, 0) of the
    ! hand workspace, then set PP to routine rHandFollow in both tasks and run.
    !***********************************************************************

    CONST string SERVER_IP := "192.168.125.1";
    CONST num PORT_L := 1025;
    CONST num PORT_R := 1026;
    ! Name of the plane work object in each task (read at run time).
    CONST string WOBJ_NAME_L := "wobjPlaneXY";
    CONST string WOBJ_NAME_R := "wobjPlaneXY1";

    ! Hand workspace in the app is X +-200 mm, Y +-150 mm.
    CONST num LIMIT_X := 200;
    CONST num LIMIT_Y := 150;
    ! TRUE : center = position of the arm when rHandFollow starts.
    ! FALSE: center = origin (0, 0) of wobjPlaneXY, Z kept from the start pose.
    CONST bool START_IS_CENTER := TRUE;
    ! Ignore changes smaller than this (mm) to avoid jitter.
    CONST num DEADBAND := 2;
    ! Seconds without a reply before the connection is reset.
    CONST num RECV_TIMEOUT := 3;
    CONST speeddata vFollow := [150, 200, 5000, 1000];
    CONST zonedata zFollow := z5;

    ! Adapt these values to the fitted fingers, configured physical limit
    ! and workpiece. g_MoveTo opens without pushing against the end stop.
    CONST num GRIP_OPEN_MM := 24;
    CONST num GRIP_FORCE_N := 5;

    TASK PERS tooldata tFollow := [TRUE, [[0, 0, 0], [1, 0, 0, 0]], [0.001, [0, 0, 0.001], [1, 0, 0, 0], 0, 0, 0]];
    TASK PERS wobjdata wFollow := [FALSE, TRUE, "", [[0, 0, 0], [1, 0, 0, 0]], [[0, 0, 0], [1, 0, 0, 0]]];

    VAR socketdev server;
    VAR socketdev client;
    VAR robtarget pCenter;
    VAR num lastX;
    VAR num lastY;
    VAR num lastGrip := -1;

    PROC rHandFollow()
        VAR num port;
        VAR wobjdata wPlane;

        ! GetDataVal needs a VAR argument, so read into wPlane first.
        IF GetTaskName() = "T_ROB_L" THEN
            port := PORT_L;
            GetDataVal WOBJ_NAME_L, wPlane;
        ELSE
            port := PORT_R;
            GetDataVal WOBJ_NAME_R, wPlane;
        ENDIF
        wFollow := wPlane;

        ! Keep the tool that is active now, and its orientation, arm angle and
        ! configuration; only X and Y in the plane work object change.
        tFollow := CTool();
        ConfL \Off;
        pCenter := CRobT(\Tool:=tFollow \WObj:=wFollow);
        IF NOT START_IS_CENTER THEN
            pCenter.trans.x := 0;
            pCenter.trans.y := 0;
        ENDIF

        SocketClose client;
        SocketClose server;
        SocketCreate server;
        SocketBind server, SERVER_IP, port;
        SocketListen server;

        WHILE TRUE DO
            TPWrite GetTaskName() + ": waiting for laptop on port " + NumToStr(port, 0);
            SocketAccept server, client \Time:=WAIT_MAX;
            TPWrite GetTaskName() + ": laptop connected";
            ! Start from the current position, not from the last target.
            lastX := 9999;
            lastY := 9999;
            lastGrip := -1;
            FollowLoop;
        ENDWHILE
    ENDPROC

    PROC FollowLoop()
        VAR string line;
        VAR num valid;
        VAR num x;
        VAR num y;
        VAR num grip;
        VAR robtarget pTarget;

        WHILE TRUE DO
            SocketSend client \Str:="R,G\0A";
            line := ReadLine();
            IF ParseTarget(line, valid, x, y, grip) AND valid = 1 THEN
                ! Independent of XY deadband: a stationary wrist can still
                ! open/close the gripper attached to this task's robot arm.
                ApplyGripper grip;
                x := Clamp(x, LIMIT_X);
                y := Clamp(y, LIMIT_Y);
                IF Abs(x - lastX) >= DEADBAND OR Abs(y - lastY) >= DEADBAND THEN
                    pTarget := pCenter;
                    pTarget.trans.x := pCenter.trans.x + x;
                    pTarget.trans.y := pCenter.trans.y + y;
                    MoveL pTarget, vFollow, zFollow, tFollow \WObj:=wFollow;
                    lastX := x;
                    lastY := y;
                ELSE
                    WaitTime 0.02;
                ENDIF
            ELSE
                lastGrip := -1;
                WaitTime 0.02;
            ENDIF
        ENDWHILE
    ERROR
        IF ERRNO = ERR_SOCK_CLOSED OR ERRNO = ERR_SOCK_TIMEOUT THEN
            TPWrite GetTaskName() + ": laptop disconnected";
            lastGrip := -1;
            SocketClose client;
            RETURN;
        ENDIF
        ! Preserve SmartGripper/calibration and motion errors for the caller.
        RAISE;
    ENDPROC

    PROC ApplyGripper(num grip)
        IF grip = -1 THEN
            ! Re-acquisition of a stable sign must send a fresh command.
            lastGrip := -1;
            RETURN;
        ENDIF
        IF grip = lastGrip RETURN;
        IF grip = 1 THEN
            g_MoveTo GRIP_OPEN_MM, \NoWait;
        ELSEIF grip = 0 THEN
            g_GripIn \holdForce:=GRIP_FORCE_N, \NoWait;
        ENDIF
        ! This remembers a command, not confirmation of physical position.
        lastGrip := grip;
    ENDPROC

    FUNC string ReadLine()
        VAR string buffer := "";
        VAR string chunk;

        WHILE StrFind(buffer, 1, "\0A") > StrLen(buffer) DO
            SocketReceive client \Str:=chunk \Time:=RECV_TIMEOUT;
            buffer := buffer + chunk;
        ENDWHILE
        RETURN buffer;
    ENDFUNC

    FUNC bool ParseTarget(string line, INOUT num valid, INOUT num x, INOUT num y, INOUT num grip)
        VAR num c1;
        VAR num c2;
        VAR num c3;
        VAR num endPos;
        VAR string ending;

        grip := -1;
        endPos := StrFind(line, 1, "\0D\0A");
        IF endPos > StrLen(line) RETURN FALSE;
        ending := StrPart(line, endPos, StrLen(line) - endPos + 1);
        IF ending <> "\0A" AND ending <> "\0D\0A" RETURN FALSE;
        c1 := StrFind(line, 1, ",");
        IF c1 <= 1 OR c1 >= endPos RETURN FALSE;
        c2 := StrFind(line, c1 + 1, ",");
        IF c2 <= c1 + 1 OR c2 >= endPos RETURN FALSE;
        c3 := StrFind(line, c2 + 1, ",");
        IF NOT StrToVal(StrPart(line, 1, c1 - 1), valid) RETURN FALSE;
        IF valid <> 0 AND valid <> 1 RETURN FALSE;
        IF NOT StrToVal(StrPart(line, c1 + 1, c2 - c1 - 1), x) RETURN FALSE;
        IF c3 < endPos THEN
            IF c3 <= c2 + 1 OR c3 >= endPos - 1 RETURN FALSE;
            IF StrFind(line, c3 + 1, ",") < endPos RETURN FALSE;
            IF NOT StrToVal(StrPart(line, c2 + 1, c3 - c2 - 1), y) RETURN FALSE;
            IF NOT StrToVal(StrPart(line, c3 + 1, endPos - c3 - 1), grip) RETURN FALSE;
            IF grip <> -1 AND grip <> 0 AND grip <> 1 RETURN FALSE;
        ELSE
            IF endPos <= c2 + 1 RETURN FALSE;
            IF NOT StrToVal(StrPart(line, c2 + 1, endPos - c2 - 1), y) RETURN FALSE;
        ENDIF
        RETURN TRUE;
    ENDFUNC

    FUNC num Clamp(num value, num limit)
        IF value > limit RETURN limit;
        IF value < -limit RETURN -limit;
        RETURN value;
    ENDFUNC
ENDMODULE
