def main():
    choice = input("Press and enter 1 for code about a conversation, press and enter 2 for code about days of the week, press and enter 3 for code about colors, or press and enter 4 for code about probability")
    if choice == "1":
        print("hi! :)")
        name = input("What is your name?") #promt the user, put reply in username
        k = input("How are you,"+name+"?") #promt the user, put reply in username
        print("That's good!")
        x = float(input("How many cents do you have?")) #promt the user, put reply in username
        print("You have", x, "cents!")
        print(x//25, "are quarters")
        print(x%25//10, "are dimes")
        print(x%25%10//5, "are nickles")
        print(x%5//1, "are pennies")
    elif choice == "2":
        x = input("Enter a day of the week:")
        if x== "Sunday":
            print("Monday")
        if x== "Monday":
            print("Tuesday")
        if x== "Tuesday":
            print("Wednesday")
        if x== "Thursday":
            print("Friday")
        if x== "Friday":
            print("Saturday")
        if x== "Saturday":
            print("Sunday")
    elif choice == "3":
        x = input("Give me a color:")
        if x== "Red":
            print("in spanish it is: Rojo")
        if x== "Orange":
            print("in spanish it is: Naranja")
        if x== "Yellow":
            print("in spanish it is: Amarillo")
        if x== "Green":
            print("in spanish it is: Verde")
        if x== "Purple":
            print("in spanish it is: Morado")
        if x== "Blue":
            print("in spanish it is: Azul")
        if x== "Black":
            print("in spanish it is: Negro")
        if x== "Grey":
            print("in spanish it is: Gris")
        if x== "Pink":
            print("in spanish it is: Rosa")
    elif choice == "4":
        import random

        print("You are going to flip two coins and one 6 sided dice!")


        def flipcoin():
            result = ("Heads", "Tails")
            result = random.choice(result)
            return result

        resultA = flipcoin()
        print("Your coin landed:", resultA)



        resultB = flipcoin()
        print("Your coin landed:", resultB)

        def rolldice():
            result = random.randrange(1, 7)
            return result
        result = rolldice()
        print("Your random dice number is:", result)

        for i in range(6):
            # Flip two coin    
            resultA = flipcoin()
            resultB = flipcoin()

            # Roll the dice
            dice_result = rolldice()

            # Print results
            print("Flip 1:", resultA, "! Flip 2:", resultB, "! Dice roll:", dice_result)
    else:
        print("Invalid input")