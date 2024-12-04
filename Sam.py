def main():
    if input("Press and enter 1 for code about rolling two dice, press and enter 2 for code about guessing elements") == "1":
        #Monte Carlo for simulating distribution outcomes when rolling 2d6

        import random

        step = 0
        correct2 = 0
        correct3 = 0
        correct4 = 0
        correct5 = 0
        correct6 = 0
        correct7 = 0
        correct8 = 0
        correct9 = 0
        correct10= 0
        correct11 = 0
        correct12 = 0

        print("This code uses a monte carlo method to roll probabilities of 2 dice equaling a given result, and rolls 2d6 100 times and then 100,000 times to see which roll has a more accurate distribution, when compared to the odds of a given pair of 2 six sided dice adding up to a given result. For each set of rolls, the % of the rolls that was a given number was shown, along with the statistical % of the rolls that should equal said number.")

        print("Lets roll 100 times.")


        for n in range(100):
            step+=1
            d1 = random.randrange(1,7)
            d2 = random.randrange(1,7)
        #  print(d1,d2)
            if d1+d2==7:
                correct7 += 1
            elif d1+d2==6:
                correct6 +=1
            elif d1+d2==8:
                correct8 +=1
            elif d1+d2==5:
                correct5 +=1
            elif d1+d2==9:
                correct9 +=1
            elif d1+d2==4:
                correct4 +=1
            elif d1+d2==10:
                correct10 +=1
            elif d1+d2==3:
                correct3 +=1
            elif d1+d2==11:
                correct11 +=1
            elif d1+d2==2:
                correct2 +=1
            elif d1+d2==12:
                correct12 +=1

        print("You got 2s", correct2*100/step,"% of the time")
        print("Experimentally it should be", 2.77,"% of the time")

        print("You got 3s", correct3*100/step,"% of the time")
        print("Experimentally it should be", 5.55,"% of the time")

        print("You got 4s", correct4*100/step,"% of the time")
        print("Experimentally it should be", 8.33,"% of the time")

        print("You got 5s", correct5*100/step,"% of the time")
        print("Experimentally it should be", 11.11,"% of the time")

        print("You got 6s", correct6*100/step,"% of the time")
        print("Experimentally it should be", 13.88,"% of the time")

        print("You got 7s", correct7*100/step,"% of the time")
        print("Experimentally it should be", 16.66,"% of the time")

        print("You got 8s", correct8*100/step,"% of the time")
        print("Experimentally it should be", 13.88,"% of the time")

        print("You got 9s", correct9*100/step,"% of the time")
        print("Experimentally it should be", 11.11,"% of the time")

        print("You got 10s", correct10*100/step, "% of the time")
        print("Experimentally it should be", 8.33,"% of the time")

        print("You got 11s", correct11*100/step,"% of the time")
        print("Experimentally it should be", 5.55,"% of the time")

        print("You got 12s", correct12*100/step,"% of the time")
        print("Experimentally it should be", 2.77,"% of the time")

        print("Lets roll 100,000 times.")


        for n in range(100000):
            step+=1
            d1 = random.randrange(1,7)
            d2 = random.randrange(1,7)
        #  print(d1,d2)
            if d1+d2==7:
                correct7 += 1
            elif d1+d2==6:
                correct6 +=1
            elif d1+d2==8:
                correct8 +=1
            elif d1+d2==5:
                correct5 +=1
            elif d1+d2==9:
                correct9 +=1
            elif d1+d2==4:
                correct4 +=1
            elif d1+d2==10:
                correct10 +=1
            elif d1+d2==3:
                correct3 +=1
            elif d1+d2==11:
                correct11 +=1
            elif d1+d2==2:
                correct2 +=1
            elif d1+d2==12:
                correct12 +=1

        print("You got 2s", correct2*100/step,"% of the time")
        print("Experimentally it should be", 2.77,"% of the time")

        print("You got 3s", correct3*100/step,"% of the time")
        print("Experimentally it should be", 5.55,"% of the time")

        print("You got 4s", correct4*100/step,"% of the time")
        print("Experimentally it should be", 8.33,"% of the time")

        print("You got 5s", correct5*100/step,"% of the time")
        print("Experimentally it should be", 11.11,"% of the time")

        print("You got 6s", correct6*100/step,"% of the time")
        print("Experimentally it should be", 13.88,"% of the time")

        print("You got 7s", correct7*100/step,"% of the time")
        print("Experimentally it should be", 16.66,"% of the time")

        print("You got 8s", correct8*100/step,"% of the time")
        print("Experimentally it should be", 13.88,"% of the time")

        print("You got 9s", correct9*100/step,"% of the time")
        print("Experimentally it should be", 11.11,"% of the time")

        print("You got 10s", correct10*100/step, "% of the time")
        print("Experimentally it should be", 8.33,"% of the time")

        print("You got 11s", correct11*100/step,"% of the time")
        print("Experimentally it should be", 5.55,"% of the time")

        print("You got 12s", correct12*100/step,"% of the time")
        print("Experimentally it should be", 2.77,"% of the time")
    else:
        #!/usr/bin/env python3
        # -*- coding: utf-8 -*-
        """
        Created on Wed Nov  6 11:04:43 2024

        @author: samgelber
        """

        #!/usr/bin/env python3
        # -*- coding: utf-8 -*-
        """
        Created on Mon Nov  4 11:06:03 2024

        @author: samgelber
        """

        print("In this code, you will choose an element and the code will guess it.")
        print("Go to https://tinyurl.com/yjt4vvex and then think of an element. Also open https://tinyurl.com/2bhyhjfw. Go to https://tinyurl.com/bdhzcppc for a chart of unpaired electrons for each element.")
        print("This code is unfinished. Say that your element is a transition metal.")
        q1=input("Is your element a form of metal, not including a metalloid/semimetal?")
        if q1 == "yes":
            q2=input("Is your element a transition metal?")
            if q2 == "yes":
                q3=input("Is your element in the 3d row?")
                if q3 == "yes":
                    q4=input("Does your element have 1 unpaired electron?")
                    if q4 == "yes":
                        q5=input("Is your element scandium?")
                        if q5 == "yes":
                            print("Great, I guessed your element.")
                        else:
                            print("Your element is copper.")
                    else:
                        q6=input("Does your element have 2 unpaired electrons?")
                        if q6 == "yes":
                            q7=input("Is your element titanium?")
                            if q7 == "yes":
                                print("Great, I guessed your element.")
                            else:
                                print("Your element is nickle.")
                        else:
                            q7=input("Does your element have 3 unpaired electrons?")
                            if q7 == "yes":
                                q8=input("Is your element vanadium?")
                                if q8 == "yes":
                                    print("Great, I guessed your element")
                                else:
                                    print("Your element is cobalt.")
                            else:
                                q8=input("Does your element have 4 unpaired electrons?")
                                if q8 == "yes":
                                    q9=input("Is your element chromium?")
                                    if q9 == "yes":
                                        print("Great, I guessed your element!")
                                    else:
                                        print("Your element is iron!")
                                else:
                                    q9=input("Does your element have 5(or more) unpaired electrons?")
                                    if q9 == "yes":
                                        print("Your element is manganese.")
                                    else:
                                        print("Your element is zinc.")    
                else: 
                    q4=input("Is your element in the 4d row?")
                    if q4 == "yes":
                        print("This code is unfinished. To see what a finished version would look like, rerun this code and say that your element is in the 3d row.")  
                    else:
                        q5=input("Is your element in the 5d row?")
                        print("This code is unfinished. To see what a finished version would look like, rerun the code and say that your element is in the 3d row.")
            else:
                q3=input("Is your element a lathanide?")
                if q3 =="yes":
                    print("Great, I guessed your category.")
                else:
                    q4=input("Is your element an actinide?")
                    if q4 == "yes":
                        print("Great, I guessed your category.")
                    else:
                        q5=input("Is your element a basic metal?")
                        if q5 == "yes":
                            print("Great, I guessed your category.")
                        else:
                            q6=input("Is your element from the alkaline earth category?")
                            if q6 == "yes":
                                print("Great, I guessed your category.")
                            else:
                                print("Your element is an alkali metal.")
                
        else:
            q2=input("Is your element a semimetal?")
            if q2 == "yes":
                print("Great, I guessed your element category.")
            else:
                q3=print("Is your element a nonmetal?")
                if q3 == "yes":
                    print("Great, I guessed your element category.")
                else:
                    q4=print("Is your element a halogen?")
                    if q4 == "yes":
                        print("Great, I guessed your element category.")
                    else:
                        print("Your element is a noble gas.")