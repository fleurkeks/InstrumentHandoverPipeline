MODULE MainModule
	VAR socketdev serversocket;
	VAR socketdev clientsocket;
    
	VAR string msg:= "";

	PERS wobjdata campos:= [FALSE,TRUE,"",[[0,0,0],[0,0,0,0]],[[0,0,0],[0,0,0,0]]];

	


	PROC main()
        TPWrite "hej";
        
		SocketCreate serversocket;
		SocketBind serversocket, "localhost", 1025;
		SocketListen serversocket;
		SocketAccept serversocket, clientsocket;
		
		WHILE true DO
			SocketReceive clientsocket\Str:=msg;
		    TPWrite msg;
		ENDWHILE
	ENDPROC




ENDMODULE