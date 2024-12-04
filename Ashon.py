def main():
    Board = ["□","□","□","□","□","□","□","□","□"]

    Won = False
    Player_Symbols = {1:"X",2:"O"}

    L_Mapper = {"A":0,"B":1,"C":2}

    Player_Cycle = 1

    def Show_Board():
        #Input: Null
        #Action: Formats the tic-tac-toe board into a viewable format
        #Output: returns a printed game board!
        index = 0
        
        print("  A B C")
        for y in range(3):
            row = str(y + 1)
            row += " "
            
            for x in range(3):
    
                row += Board[index]
                row += " "
                index += 1
        
            print(row)
        print("_______________________________")
        

    def Player_Move(PlayerNumber):

        print("It is player ",str(PlayerNumber), "'s turn!!!")
        
        Location = input("Where would You like to place your token?")
        
        if not len(Location) == 2:
            print("PLEASE INPUT A 2 CHARACTER LOCATION (EX: A1,B2,C3)","\n\n\n")
            Player_Move(PlayerNumber)
        
        elif Location[0].upper() in L_Mapper.keys() and Location[1].isnumeric():
            if int(Location[1]) > 0 and int(Location[1]) < 4:
                print("YAY!")
                Mapper(Location,PlayerNumber)
            else:
                print("Please choose a position within A1 and C3","\n\n\n")
                Player_Move(PlayerNumber)
        else:
            print("Please Input an existing Collumn Identifier (A,B,C)","\n\n\n")
            Player_Move(PlayerNumber)        

            
    def Switch_Players(Current_Player):
        
        if Current_Player == 1:
            return 2
        elif Current_Player == 2:
            return 1

    def Mapper(L,Player):
        
        #Input: String
        #Acton: Convert a String into an index in the list 'Board'
        #Output: Return index as an int
        
        Board_Index = L_Mapper[L[0]]
        Board_Index += (int(L[1])-1)*3
        #Board_Index += int(L[1])
        
        print(Board_Index)
        
        if not Board[Board_Index] == "□":
            print("This Cell has already been played!!! \n\nChoose a new Cell")
            Show_Board()
            Player_Move(Player)
        else:
            Board[Board_Index] = Player_Symbols[Player]
            Show_Board()
            Board_Scanner()
            Player_Move(Switch_Players(Player))
            

    def Board_Scanner():
        
        #Scans Vertically
        Index = 0
        
        for l in range(3):
            Index = l - 1
            Result = ""
            for c in range(3):
                Result += Board[Index]
                Index += 3
                
                
            if Check_For_Win(Result) == True:
                Replay = input("Would you like to play again?")
                
                if Replay == "y":
                    pass
                else:
                    quit()
                    
        #Scans for Horizontal
                    
        for w in range(3): 
            Index = 0 + ((w-1)*3)
            Result = ""
            for c in range(3):
                Result += Board[Index]
                Index += 1
                
                
            if Check_For_Win(Result) == True:
                Replay = input("Would you like to play again?")
                
                if Replay == "y":
                    pass
                else:
                    quit()
        
        #Scans for Diagonals
                    
        Result = Board[6] + Board[4] + Board[2]
        if Check_For_Win(Result) == True:
            Replay = input("Would you like to play again?")
            
            if Replay == "y":
                pass
            else:
                quit()

    
        Result = Board[8] + Board[4] + Board[0]
        if Check_For_Win(Result) == True:
            Replay = input("Would you like to play again?")
            
            if Replay == "y":
                pass
            else:
                quit()

        
    def Check_For_Win(S):
        
        #Input: Takes in a String with tokens on given cells (ex: "- - x" or "o o o "
        #Action: Determines if the string has the same character 3 times, and determines if a win has occurred
        #Output: Return True or False depending on win or no win!
        
                    
        if S == "XXX" or S == "OOO":
            print("\n\n\n\n\nPLAYER HAS WON!!!")
    
            return True
        else:
            return False

    #  A B C
    #1 X X □ 
    #2 □ O □ 
    #3 □ O X 


    Show_Board()
    Player_Cycle = 1
    Player_Move(1)