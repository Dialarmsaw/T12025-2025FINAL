def main():
    import random

    def weirdCardThing():
        print("This code runs a simulation for the following question: ")
        print("If you shuffle a deck, remove 8 cards, then cut the deck in half, what are the odds that 4 any specific number in either of the halves")
        n = int(input("How many times do you want to run the simulation (no more than 500,000 recommended based on time)? "))
        numSets = []
        for trial in range(n):
            if (trial * 100) % n == 0:
                print(str(int(trial/n*100)) + "% done", end = "\r")
            # Make deck
            deck = []
            for i in range(13):
                for j in range(4):
                    deck.append(i)
            # Shuffle deck
            shuffledDeck = []
            for i in range(52):
                shuffledDeck.append(deck.pop(random.randint(0,51-i)))
            # Remove 8 cards
            for i in range(8):
                shuffledDeck.pop(0)
            # Check for sets of 4
            sets = 0
            for i in range(1):
                counter = 0
                for j in range(22):
                    if shuffledDeck[j] == i:
                        counter += 1
                if counter == 4:
                    sets += 1
                counter = 0
                for j in range(22,44):
                    if shuffledDeck[j] == i:
                        counter += 1
                if counter == 4:
                    sets += 1        
            numSets.append(sets)
        print()
        print("Mathematically derived correct answer (not simulated): 0.0537308")
        print("Estimated chance based on simulation: " + str(sum(numSets)/len(numSets)))

    def base10ToBinaryRunner():
        num = float(input("Give me a number (no fractions, decimal place allowed): "))
        p = -1 * int(input("How many decimal places do you want (0 for none)? "))
        print("This function is recursive, meaning that it calls itself to solve the problem. Print statements will be used to illustrate what it is doing. ")
        print()
        print("The output is: " + base10ToBinary(num, p))

    def base10ToBinary(number, power):
        if number - 2**power < 0:
            print("The function found that 2^" + str(power+1) + " is more than the number, and returns \"\"")
            return ""
        print("The function calls itself for a power one higher (" + str(power+1) + ") and adds " + str(int(2**(((number%(2**(power+1)))/2**power)-1))) + " to its end")
        return base10ToBinary(number, power + 1) + ("." * int(1/2**abs(power+1))) + str(int(2**(((number%(2**(power+1)))/2**power)-1)))

    def base10ToBinaryNoPrint(number, power):
        if number - 2**power < 0:
            return ""
        return base10ToBinaryNoPrint(number, power + 1) + ("." * int(1/2**abs(power+1))) + str(int(2**(((number%(2**(power+1)))/2**power)-1)))

    def makePassengers():
        if popIncrease < 1:
            if random.randrange(int(1/popIncrease)+1) == 0:
                floorInfo[random.randrange(numFloors)].append([random.randrange(numFloors), 0])
            for i in range(len(floorInfo)):
                j = 0
                while j < len(floorInfo[i]):
                    if floorInfo[i][j][0] == i:
                        del floorInfo[i][j]
                    else:
                        j += 1
        else: 
            for i in range(int(popIncrease)):
                floorInfo[random.randrange(numFloors)].append([random.randrange(numFloors), 0])
            for i in range(len(floorInfo)):
                j = 0
                while j < len(floorInfo[i]):
                    if floorInfo[i][j][0] == i:
                        del floorInfo[i][j]
                    else:
                        j += 1

    def moveElevators(rule):
        if rule == 0:
            # Moves in the direction of the average floor, opening the first chance it gets
            if sum([j[0] for j in elevators[i][2:]]) / len(elevators[i][2:]) - elevators[i][1] > 0:
                elevators[i][1] += 1
            else:
                elevators[i][1] -= 1
            for j in range(2, len(elevators[i])):
                if elevators[i][j][0] == elevators[i][1]:
                    elevators[i][0] = True
        if rule == 1:
            if random.randrange(2) == 0:
                if elevators[i][1] < numFloors:
                    elevators[i][1] += 1
            else: 
                if elevators[i][1] > 0:
                    elevators[i][1] -= 1
            for j in range(2, len(elevators[i])):
                if elevators[i][j][0] == elevators[i][1]:
                    elevators[i][0] = True
        if rule == 2:
            elevators[i][1] += elevatorDirections[i]
            if elevators[i][1] == 0: 
                elevatorDirections[i] = 1
            if elevators[i][1] == numFloors - 1:
                elevatorDirections[i] = -1
            for j in range(2, len(elevators[i])):
                if elevators[i][j][0] == elevators[i][1]:
                    elevators[i][0] = True
            if floorInfo[elevators[i][1]] != []:
                elevators[i][0] = True
            
    choice = input("Input 1 for a Monte Carlo simulation of a statistics problem, 2 for a recursive conversion of a number into binary, or 3 for a simulation of various strategies of elevator movement: ")
    print()

    if choice == "1":
        weirdCardThing()

    if choice == "2":
        base10ToBinaryRunner()

    if choice == "3":
        # Getting numeric info from person running code
        numFloors = int(input("How many floors should there be? "))
        numElevators = int(input("How many elevators should there be? "))
        capacity = int(input("What is the capacity of an elevator? "))
        popIncrease = float(input("How many people appear each timestep? ")) * (1 + 1/numFloors)
        runs = int(input("How many timesteps do you want to run this for? "))
        numTrials = int(input("How many trials per rule? "))
        rules = [0, 1, 2]
        waitTimes = []
        for i in range(len(rules)):
            waitTimes.append([])
            for j in range(numTrials):
                waitTimes[i].append([])
        # Go
        for chosenRule in rules:
            for trial in range(numTrials):
                # Make building
                floorInfo = []
                elevators = []
                elevatorDirections = []
                for i in range(numFloors):
                    floorInfo.append([])
                for i in range(numElevators):
                    elevators.append([True, 0])
                    elevatorDirections.append(1)
                # Run
                for timestep in range(runs):
                    makePassengers()
                    for i in range(len(elevators)):
                        if elevators[i][0]:
                            for j in range(5):
                                # Attempt to disembark
                                k = 2
                                while k < len(elevators[i]) and elevators[i][k][0] != elevators[i][1]:
                                    k += 1
                                if k < len(elevators[i]):
                                    waitTimes[chosenRule][trial].append(elevators[i][k][1])
                                    del elevators[i][k]
                                # Attempt to add passengers
                                elif (not len(elevators[i]) == capacity + 2) and floorInfo[elevators[i][1]] != []:
                                    elevators[i].append(floorInfo[elevators[i][1]].pop(0))
                                # Shut the doors
                                elif len(elevators[i]) > 2:
                                    elevators[i][0] = False
                        else:
                            moveElevators(chosenRule)
                    # Increas wait times
                    for i in range(len(floorInfo)):
                        for j in range(len(floorInfo[i])):
                            floorInfo[i][j][1] += 1
                    for i in range(len(elevators)):
                        for j in range(2, len(elevators[i])):
                            elevators[i][j][1] += 1
        for i in range(len(waitTimes)):
            for j in range(len(waitTimes[i])):
                waitTimes[i][j] = sum(waitTimes[i][j]) / len(waitTimes[i][j])
        print()
        print("Rule 0 is elevators moving to the average desired floor of its passengers")
        print("Rule 1 is elevators chosing to move up or down randomly")
        print("Rule 2 is elevators moving from the top floor to the bottom floor and back up forever")
        print()
        for i in range(len(waitTimes)):
            print("Rule " + str(i) + " average average completed journy time: " + str((sum(waitTimes[i]) / len(waitTimes[i]))))

main()