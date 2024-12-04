def main():
    if input("1 for blackjack and 2 for deck simulator (question 10).") == "1":
        import random #Plays Blackjack
        from random import shuffle

        Deck = []
        Hand = []
        DealerHand = []
        Counts = []
        Totals = [0,0]

        def Wipe(): #Wipes the deck clearing hands and remaking the deck
            Deck.clear()
            Hand.clear()
            for i in range(4): #creates "hands" the player could have
                Hand.append([])
            print(Hand)
            Counts.clear()
            for i in range(13): #adding all cards to the deck
                Counts.append(0)
            Deck.append("Ace of Diamonds")
            Deck.append("Ace of Clubs")
            Deck.append("Ace of Hearts")
            Deck.append("Ace of Spades")
            for i in range(1,10):
                Deck.append(str(i+1)+" of Diamonds")
                Deck.append(str(i+1)+" of Clubs")
                Deck.append(str(i+1)+" of Hearts")
                Deck.append(str(i+1)+" of Spades")
            Deck.append("Jack of Diamonds")
            Deck.append("Jack of Clubs")
            Deck.append("Jack of Hearts")
            Deck.append("Jack of Spades")
            Deck.append("Queen of Diamonds")
            Deck.append("Queen of Clubs")
            Deck.append("Queen of Hearts")
            Deck.append("Queen of Spades")
            Deck.append("King of Diamonds")
            Deck.append("King of Clubs")
            Deck.append("King of Hearts")
            Deck.append("King of Spades")

        def RunIt(): #Wipes then shuffles the deck
            Wipe()
            shuffle(Deck)

        def StartRound(): #Runs the first round of blackjack (round includes any action such as splitting or standing)
            DrawNum = 0
            RunIt()
            for i in range(2):
                Hand[0].append(Deck[0])
                Deck.remove(Deck[0])
            for i in range(2):
                DealerHand.append(Deck[0])
                Deck.remove(Deck[0])
            print("Your Hand:",Hand)
            print("Dealer's Hand:",DealerHand[1])
            print("What will you do?","\n","1 for hit","\n","2 for split")
            Move = int(input(" 3 for stand "))
            if Move == 1:
                SplitHand = int(input("Which hand will you hit? "))
                hitMe(SplitHand-1)
            elif Move == 2:
                SplitHand = int(input("Which hand will you split? "))
                Split(SplitHand-1)
            else:
                DealerDraws()

        def Round(): #Runs a normal round of blackjack (anything after the first run)
            print("Your Hand:",Hand)
            print("Dealer's Hand:",DealerHand[1]) 
            print("What will you do?","\n","1 for hit","\n","2 for split")
            Move = int(input(" 3 for stand "))
            if Move == 1:
                SplitHand = int(input("Which hand will you hit? "))
                hitMe(SplitHand-1)
            elif Move == 2:
                SplitHand = int(input("Which hand will you split? "))
                Split(SplitHand-1)
            else:
                DealerDraws()

        def Value(DHand): #Checks the value of a single hand of cards
            Total = 0
            for i in range(len(DHand)):
                s = DHand[i] #takes the first letter/symbol of the card
                s = s[:1]
                for j in range(1,10): #evaluates the card
                    if s == str(j):
                        if s == str(1):
                            Total += 10
                        else:
                            Total += j
                if s == "J" or s == "Q" or s == "K":
                    Total += 10
                elif s == "A":
                    Total += 11 #adds all totals together
            if Total > 21: #evaluates total
                for i in range(len(DHand)):
                    s = DHand[i]
                    s = s[:1]
                    if s == "A":
                        Total -= 10
            return Total
            
        def Bust(SplitHand,CHand): #Evaluates a hand to see whether it has bust or not (only for player)
            Total = 0
            for i in range(len(CHand[SplitHand])):
                s = CHand[SplitHand][i]
                s = s[:1]
                for j in range(1,10):
                    if s == str(j):
                        if s == str(1):
                            Total += 10
                        else:
                            Total += j
                if s == "J" or s == "Q" or s == "K":
                    Total += 10
                elif s == "A":
                    Total += 11
            if Total > 21:
                for i in range(len(CHand[SplitHand])):
                    s = CHand[SplitHand][i]
                    s = s[:1]
                    if s == "A":
                        Total -= 10
            if Total > 21:
                print("You have bust at",Total,"You lose.")
                return 2
            elif Total == 21:
                print("Blackjack!")
                return 1
            else:
                print("Total:",Total)
                return 0
            

        def DealerBust(CHand): #Evaluates a hand to see whether it has bust or not (only for dealer)
            Total = 0
            for i in range(len(CHand)):
                s = CHand[i]
                s = s[:1]
                for j in range(1,10):
                    if s == str(j):
                        if s == str(1):
                            Total += 10
                        else:
                            Total += j
                if s == "J" or s == "Q" or s == "K":
                    Total += 10
                elif s == "A":
                    Total += 11
            if Total > 21:
                for i in range(len(CHand)):
                    s = CHand[i]
                    s = s[:1]
                    if s == "A":
                        Total -= 10
                if Total > 21:
                    print("Dealer has bust at",Total,".")
                    return 2
                elif Total == 21:
                    print("Blackjack!")
                    return 1
                else:
                    print("Total:",Total)
                    return 0
                    
            elif Total == 21:
                print("Blackjack!")
                return 1
            else:
                print("Total:",Total)
                return 0

        def hitMe(SplitHand): #Hits players hand to add another card
            Hand[SplitHand].append(Deck[0])
            Deck.remove(Deck[0])
            if Bust(SplitHand, Hand) < 2:
                Round()
            
        def Split(SplitHand): #splits players hand
            s1 = Hand[SplitHand][0]
            s1 = s1[:1]
            s2 = Hand[SplitHand][1]
            s2 = s2[:1]
            if len(Hand[SplitHand]) > 2: #evaluates to see if cards are the same
                print("You cannot split something higher than a pair.")
            else: #if cards are the same you can split
                openHand = 1
                Op = True
                inte = 1
                while Op: #iterates to find open hand
                    if len(Hand[inte]) == 0:
                        openHand = inte
                        Op = False
                    inte += 1
                if s1 == s2:
                    Hand[SplitHand+openHand].append(Hand[SplitHand][1])
                    Hand[SplitHand][1] = Deck[0]
                    Deck.remove(Deck[0])
                    Hand[SplitHand+openHand].append(Deck[0])
                    Deck.remove(Deck[0])
                    
                else:
                    print("Hand",SplitHand+1,"is not a pair you can split.")
            Round()
            
        def DealerDraws(): 
            while Value(DealerHand) < 17:
                DealerHand.append(Deck[0])
                Deck.remove(Deck[0])
            if DealerBust(DealerHand) < 2:
                Total = Value(DealerHand)
                TotalPlay = [0,0,0,0]
                for i in range(len(TotalPlay)):
                    if len(Hand[i]) >= 2:
                        for j in range(len(Hand[i])):
                            TotalPlay[i] += Value(Hand[i][j])
                for i in range(len(TotalPlay)):
                    if Value(DealerHand) > TotalPlay[i] and TotalPlay[i] != 0:
                        print("You lost hand",i+1,".")
                    elif Value(DealerHand) == TotalPlay[i] and TotalPlay[i] != 0:
                        print("You tied hand",i+1,".")
                    elif Value(DealerHand) < TotalPlay[i] and TotalPlay[i] != 0:
                        print("You won hand",i+1,".")
            else:
                print("Dealer Busted. Player wins!")
        StartRound()
    else:
        import random
        from random import shuffle

        Deck = []
        Hand = []
        Half = []
        Counts = []
        Totals = [0,0]

        def Wipe(): #wipes the deck refreshing it, also wipes hand and half
            Deck.clear()
            Hand.clear()
            Half.clear()
            Counts.clear()
            for i in range(13):
                Counts.append(0)
            Deck.append("Ace of Diamonds")
            Deck.append("Ace of Clubs")
            Deck.append("Ace of Hearts")
            Deck.append("Ace of Spades")
            for i in range(1,10):
                Deck.append(str(i+1)+" of Diamonds")
                Deck.append(str(i+1)+" of Clubs")
                Deck.append(str(i+1)+" of Hearts")
                Deck.append(str(i+1)+" of Spades")
            Deck.append("Jack of Diamonds")
            Deck.append("Jack of Clubs")
            Deck.append("Jack of Hearts")
            Deck.append("Jack of Spades")
            Deck.append("Queen of Diamonds")
            Deck.append("Queen of Clubs")
            Deck.append("Queen of Hearts")
            Deck.append("Queen of Spades")
            Deck.append("King of Diamonds")
            Deck.append("King of Clubs")
            Deck.append("King of Hearts")
            Deck.append("King of Spades")

        def RunIt(): #runs code
            Wipe()
            shuffle(Deck)
            print(Deck)
            
            for i in range(8):
                Draw = random.randint(0,51-i)
                Hand.append(Deck[Draw])
                Deck.remove(Deck[Draw])
            print(Hand)

            for i in range(22):
                Draw = random.randint(0,43-i)
                Half.append(Deck[Draw])
                Deck.remove(Deck[Draw])
            print(Half)

            for i in range(22):
                for j in range(1,10): #Checks for pairs
                    if Half[i] == (str(j+1)+" of Diamonds") or Half[i] == (str(j+1)+" of Clubs") or Half[i] == (str(j+1)+" of Hearts") or Half[i] == (str(j+1)+" of Spades"):
                        Counts[j] += 1
                if Half[i] == ("Jack of Diamonds") or Half[i] == ("Jack of Clubs") or Half[i] == ("Jack of Hearts") or Half[i] == ("Jack of Spades"):
                    Counts[10] += 1
                elif Half[i] == ("Queen of Diamonds") or Half[i] == ("Queen of Clubs") or Half[i] == ("Queen of Hearts") or Half[i] == ("Queen of Spades"):
                    Counts[11] += 1
                elif Half[i] == ("Queen of Diamonds") or Half[i] == ("King of Clubs") or Half[i] == ("King of Hearts") or Half[i] == ("King of Spades"):
                    Counts[12] += 1
                elif Half[i] == ("Ace of Diamonds") or Half[i] == ("Ace of Clubs") or Half[i] == ("Ace of Hearts") or Half[i] == ("Ace of Spades"):
                    Counts[0] += 1
            Values = False
            for i in range(13):
                if Counts[i] == 4:
                    Values = True
            if Values:
                Totals[0] += 1 #0th place is a 4 of a kind in one half
            Totals[1] += 1 #1th place is total runs

        RunIt()
        print(Counts)
        print(Totals)

main()
        





