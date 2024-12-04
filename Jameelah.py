def main():
    import random
    step = 0 #how many steps
    correct = 0 #how many times did we get an odd number
    correct1 = 0 #how many times did we get a face card 

    def draw_card():
        deck = ["Ace",2, 3, 4, 5, 6, 7, 8 ,9, 10, "Jack", "Queen", "King"]
        draw=random.choice(deck)
        #print (draw)
        return draw
    for n in range(100000):#how many times are we counting?
        step+=1#keep track of steps
        d1 = random.randrange(1,7)#roll die 1
        d2 = draw_card ()
        #print(d1,d2)
        if d1==1 or d1== 3 or d1== 5:#check if dice rolled an odd number
        
            if d2== "Ace" or d2== "Jack" or d2=="Queen" or d2== "King":
                correct1 += 1
    print("You got an odd number and a face card", correct1*100/step,"% of the time")
    print("Experimentally it should be 15% of the time")