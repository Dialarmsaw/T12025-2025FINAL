def main():
    choice = input("Press and enter 1 for code about rolling dice, press and enter 2 for code about drawring kings from a deck of cards, press and enter 3 for code about guessing numbers")
    if choice == "1":
        import random

        list1 = []
        list2 = []
        count = 0
        threek = 0
        fourk = 0
        fivek = 0

        rollnum = int(input("how many times would you like to roll the dice"))

        for i in range (rollnum):
            list1 = []
            for i in range (5):
                list1.append(random.randrange(1,7))
            for x in range (5):#3 of a kind 
                if list1.count(list1[x]) == 3:
                    threek += 1
                    break
            for q in range (5):
                if list1.count(list1[q]) == 4:
                    fourk += 1
                    break
            for w in range (5):
                if list1.count(list1[w]) == 5:
                    fivek += 1

            print("five dice were rolled", rollnum,"times")
        print("number of occurences of 3 of a kind", threek)
        print("% of roll sets that produced 3 of a kind", (threek/rollnum)*100)
        print("number of occurences of 4 of a kind",fourk)
        print("% of roll sets that produced 4 of a kind", (fourk/rollnum)*100)
        print("number of occurences of 5 of a kind", fivek)
    if choice == "2":
        import random

        range1 = 20000
        count3 = 0

        for i in range (range1):
            c1 = random.randrange(1,53)
            c2 = random.randrange(1,52)
            if c1 >= 49 and c2 >= 49:
              count3 += 1

        expected = (0.00452489*range1)

        print(count3,"out of", range1,"outcomes drew 2 kings in a row", 100*(count3/range1), "% of ourtcomes had 2 kings in a row")
        print((100*0.00452489), "% of outcomes should have had 2 kings. ")
        print("the estimated # of two king in a row outcomes was",(expected))
        print("there were", count3-expected, "more 2 king pairs than expected")
    if choice == "3":
        import random

        count = 1

        realnumber = random.randrange(1,11)
        guessnumber = random.randrange(1,11)
        diff = abs(realnumber-guessnumber)
        if diff == 8:
            print("hypothermic")
        if diff == 7:
            print("frozen")
        if diff == 6:
            print("frigid")
        if diff == 5:
            print("a little bit chilly")
        if diff == 4:
            print("lukewarm")
        if diff == 3:
            print("beach day")
        if diff == 2:
            print("perspirating waterfalls")
        if diff == 1:
            print("positively hyperthermic")
        if realnumber == guessnumber:
            print("guess was correct")

        print("guess#",count)
        print("the number guessed was",guessnumber)
        print("real number was ",realnumber)

        print()

        while realnumber != guessnumber:
            realnumber = random.randrange(1,11)
            guessnumber = random.randrange(1,11)
            count += 1
            diff = abs(realnumber-guessnumber)
            if diff == 8:
                print("hypothermic")
            if diff == 7:
                print("frozen")
            if diff == 6:
                print("frigid")
            if diff == 5:
                print("a little bit chilly")
            if diff == 4:
                print("lukewarm")
            if diff == 3:
                print("beach day")
            if diff == 2:
                print("perspirating waterfalls")
            if diff == 1:
                print("positively hyperthermic")
            if realnumber == guessnumber:
                print("guess was correct!")
            print("guess #",count)
            print("the number guessed was", guessnumber)
            print("the real number was",realnumber)
            print()
