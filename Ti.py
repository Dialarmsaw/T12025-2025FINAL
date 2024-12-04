def main():
    if input("Press and enter 1 for a game of madlibs, press and enter 2 for code about scrabble combinations") == "1":
        name1 = input ("give me a name")
        place1 = input ("give me a place")
        ptverb = input ("give me a past tense verb")
        animal = input ("give me an animal")
        propernoun = input ("give me a proper noun")
        adjective = input ("give me an adjective")
        celebrity = input ("give me a celebrity")
        noun1 = input ("give me a noun")
        adverbing = input ("give me an adverb ending in -ing")
        shape = input ("give me a shape")
        name2 = input ("give me a name")
        name3 = input ("give me a name")
        noun2 = input ("give me a noun")
        place2 = input ("give me a place")

        print("Yesterday ", name1, " went to the ", place1,". He/She/They ", ptverb, " a/n ", animal, " and then he/she/they went to ", propernoun,".  It was very ", adjective,", and then ", celebrity, " gave him/her/them a ", noun1, ".  'What a great day', he/she/they thought.  'I think tomorow I'll go ", adverbing,"' Life is a ", shape,".  The next day ", name2, " said to ", name3, " that he/she/they wanted to go to see ", noun2, " at the ", place2, sep= "")
    else:
        import random
        pain = []
        bandaid = []
        Tile = int(input("How many tiles do you have?"))
        for e in range(Tile):
            pain.append(e)
        for q in range (1000):
            random.shuffle(pain)
            #print(pain)
            if str(pain) not in bandaid:
                bandaid.append(str(pain))
        print(bandaid)
        print(len(bandaid))
