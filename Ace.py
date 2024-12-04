def main():
    if input("Press and enter 1 for code about the alphabet, press and enter 2 for code about dice rolling probabilities") == "1":
        print("Welcome to Ace's Alphabet Code. Enter a letter, and I will tell you what comes after it.")
        alph="abcdefghijklmnopqrstuvwxyz"
        letter=input("Enter letter here:\n")
        print("The letter that comes next in the alphabet is:\n", alph[(alph.index(letter)+1)%26]), "."
    else:
        import random
        results = []
        Identif = 2
        def roll():
            number = random.randint(1,6)
            results.append(number)

        def probcalc(identifier):
            percent=0
            for x in range(len(results)):
                if results[x]==identifier:
                   percent+=1 
            percent=(percent/10)*100
            return percent

        for x in range(10):
            roll()
        i = probcalc(Identif)
        print(results)
        print("During your 10 rolls, you got ",Identif," ",i,"% of the time"))
