def implement_BPE(sentence):

    tokens = []

    for i in sentence:
    if i not in tokens:
        tokens.append(i)

    vocab = set(tokens)
    print(f"Initial vocab: {vocab}")

    k=3

    for _ in range(k):
    
    freq_tokens = {}

    for char in range(len(tokens)-1):
        pair = (tokens[char], tokens[char+1])
        freq_tokens[pair] = freq_tokens.get(pair, 0)+1

    highest_freq_pair = max(freq_tokens, key = freq_tokens.get)
    new_token = ''.join(highest_freq_pair)
    vocab.add(new_token)

    i=0
    updated_tokens = []

    while i<len(tokens)-1:
        if (tokens[i],tokens[i+1])==highest_freq_pair:
        updated_tokens.append(new_token)
        i +=2
        else:
        updated_tokens.append(tokens[i])
        i +=1
    
    tokens = updated_tokens

        

    print(f"New Vocab: {vocab}")

if __name__ == "__main__":
    sentence="Nosy Komba (Malagasy pronunciation: [nusʲ ˈkuᵐba]; lit. 'island of lemurs'), also known as Nosy Ambariovato, is a small island in Madagascar, situated between the island of Nosy Be and the northwestern coast of the main island of Madagascar. Roughly circular, it rises sharply towards a plateau and the summit of Antaninaomby at the center of the island. Administered as an arrondissement of the unitary commune and district of Nosy-Be within Diana Region, the island is divided into five fokontany (villages), with Ampangorina as the main village and administrative center. The population is mainly restricted to the northern half of the island. The economy is reliant on tourism and handicrafts, supplemented by a wide range of agricultural products. Hotels and guest houses support tourists to the island, mainly on excursions from Nosy Be."
    implement_BPE(sentence)
