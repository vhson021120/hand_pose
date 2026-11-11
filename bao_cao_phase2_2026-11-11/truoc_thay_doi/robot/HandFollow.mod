MODULE HandFollow
    !***********************************************************************
    ! YuMi follows the operator's wrists sent by the laptop (run.py).
    !
    ! Load this SAME module into both T_ROB_L and T_ROB_R. Each task opens its
    ! own TCP server: T_ROB_L on PORT_L, T_ROB_R on PORT_R.
    ! Requires RobotWare option 616-1 PC Interface (socket messaging).
    ! Uses the plane work object already declared in each task:
    !   T_ROB_L -> WOBJ_NAME_L, T_ROB_R -> WOBJ_NAME_R (copied into wFollow).
    !
    ! Protocol (one request/reply per motion step, so data never piles up):
    !   robot  -> laptop : "R\0A"
    !   laptop -> robot  : "<valid>,<x_mm>,<y_mm>\0A"   e.g. "1,12.5,-40.0"
    !   valid = 1 : move to (x, y) in wobjPlaneXY, relative to the center point
    !   valid = 0 : hand lost or robot disarmed on the laptop -> hold position
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

    TASK PERS tooldata tFollow := [TRUE, [[0, 0, 0], [1, 0, 0, 0]], [0.001, [0, 0, 0.001], [1, 0, 0, 0], 0, 0, 0]];
    TASK PERS wobjdata wFollow := [FALSE, TRUE, "", [[0, 0, 0], [1, 0, 0, 0]], [[0, 0, 0], [1, 0, 0, 0]]];

    VAR socketdev server;
    VAR socketdev client;
    VAR robtarget pCenter;
    VAR num lastX;
    VAR num lastY;

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
            FollowLoop;
        ENDWHILE
    ENDPROC

    PROC FollowLoop()
        VAR string line;
        VAR num valid;
        VAR num x;
        VAR num y;
        VAR robtarget pTarget;

        WHILE TRUE DO
            SocketSend client \Str:="R\0A";
            line := ReadLine();
            IF ParseTarget(line, valid, x, y) AND valid = 1 THEN
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
                WaitTime 0.02;
            ENDIF
        ENDWHILE
    ERROR
        IF ERRNO = ERR_SOCK_CLOSED OR ERRNO = ERR_SOCK_TIMEOUT THEN
            TPWrite GetTaskName() + ": laptop disconnected";
            SocketClose client;
            RETURN;
        ENDIF
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

    FUNC bool ParseTarget(string line, INOUT num valid, INOUT num x, INOUT num y)
        VAR num c1;
        VAR num c2;
        VAR num endPos;

        c1 := StrFind(line, 1, ",");
        IF c1 >= StrLen(line) RETURN FALSE;
        c2 := StrFind(line, c1 + 1, ",");
        IF c2 >= StrLen(line) RETURN FALSE;
        endPos := StrFind(line, c2 + 1, "\0D\0A");
        IF NOT StrToVal(StrPart(line, 1, c1 - 1), valid) RETURN FALSE;
        IF NOT StrToVal(StrPart(line, c1 + 1, c2 - c1 - 1), x) RETURN FALSE;
        IF NOT StrToVal(StrPart(line, c2 + 1, endPos - c2 - 1), y) RETURN FALSE;
        RETURN TRUE;
    ENDFUNC

    FUNC num Clamp(num value, num limit)
        IF value > limit RETURN limit;
        IF value < -limit RETURN -limit;
        RETURN value;
    ENDFUNC
ENDMODULE
