def main():
  import random

  def DPair():
    deck = []
    hand = []
    Yes = 0
    Trial = 0
    for i in range(2,15):
      deck.append(["B",i])
      deck.append(["B",i])
      deck.append(["R",i])
      deck.append(["R",i])
      random.shuffle(deck)
    Ndeck = deck.copy()
    #print(deck)
    for i in range(100000):
      if len(hand) == 4:
      #print(hand)
        #if hand[0][0] == hand[1][0] and hand[0][0] == hand[2][0] and hand[0][0] == hand[3][0]:
        Yes += 1
      hand = []
      Trial += 1
      Ndeck = deck.copy()
      random.shuffle(Ndeck)
      for b in range(5):
      #print(deck)
        card = random.choice(Ndeck)
      #print(card)
        if len(hand) == 3:
          if card[0] == hand[0][0] and card[1] == hand[2][1]:
          #print(".")
            hand.append(card)
            Ndeck.remove(card)
        #print(hand, card)
        if len(hand) == 2:
          if card[0] == hand[0][0]:
            hand.append(card)
            Ndeck.remove(card)
        if len(hand) == 1:
          if card[0] == hand[0][0] and card[1] == hand[0][1]:
            hand.append(card)
            Ndeck.remove(card)
        if len(hand) == 0:
          hand.append(card)
          Ndeck.remove(card)
      #if len(hand) == 4:
    #print(hand)
    #print(deck)
    print("Number of pairs found: " + str(Yes))
    print("Number of trials: " + str(Trial))
    print("Frequency of pairs: " + str(Yes/Trial))

  def shopping():
    CheckLine = []
    CheckNum = 0
    Line = []
    ExLine = []
    Lines = [Line, ExLine]
    ILine = []
    Item = int(input("how many items do you have? "))
    CheckNum = int(input("how many checkout lines are there? "))
    ExNum = int(input("how many express lines are there? "))
    Shoppers = int(input("how many shoppers are there? "))
    print("line diagrams")
    Extot = 0
    for i in range(CheckNum):
      Line.append(0)
    for i in range(ExNum):
      ExLine.append(0)
    TotNum = ExNum+CheckNum
    #while Shoppers > 0:
    i = 0
    Old = 0
    while i < CheckNum:
      if Shoppers > 1:
        Remove = random.randint((Shoppers//CheckNum)//2, Shoppers//2)
      #if i < len(Line) < Shoppers//TotNum:
        Old = Line[i]
        Line[i] += Remove
      else:
        Remove = 1
      #if p == 0 and i < len(Line):
        Old = Line[i]
        Line[i] += Remove
      if i <= len(Line):
      #print(i, Line)
        if Old + Remove == Line[i]:
          Shoppers = Shoppers - Remove
      i += 1
    i = 0
    Old = 0
    while i < ExNum:
      if Shoppers > 1:
        Remove = random.randint((Shoppers//ExNum)//2, Shoppers//2)
      #if i < len(Line) < Shoppers//TotNum:
        if Remove < 2:
          Remove = 3
        Old = ExLine[i]
        ExLine[i] += Remove 
      else:
        Remove = 1
      #if p == 0 and i < len(Line):
        Old = ExLine[i]
        ExLine[i] += Remove
      if i <= len(ExLine):
      #print(i, Line)
        if Old + Remove == ExLine[i]:
          Shoppers = Shoppers - Remove
      i += 1
    for i in range(len(ExLine)):
      Extot += ExLine[i]
    #while Extot > 0:
    for i in range(len(ExLine)):
      if ExLine[i] == 0:
        ExLine[i] += 1
        Extot -= 1
    print(Line, ExLine)
    if Item > 15:
      for i in range(len(Line)):
        ILine.append(random.randint(16, 40)*Line[i])
        print("the wait time for checkout line "+str(i+1)+" is "+str((ILine[i]*2)/10)+" minutes")
    if Item <= 15:
        for i in range(len(ExLine)):
          ILine.append(random.randint(1, 15)*ExLine[i])
          print("the wait time for express line "+str(i+1)+" is "+str((ILine[i]*2)/10)+" minutes")

  if input("Press and enter 1 for code about drawring from a deck of cards, press and enter 2 for a simulation of shopping lines") == "1":
    DPair()
  else:
    shopping()
