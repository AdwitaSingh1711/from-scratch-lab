import torch
import torch.nn as nn

torch.manual_seed(42)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.Wh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Ux = nn.Parameter(torch.randn(input_size, hidden_size))
        self.bh = nn.Parameter(torch.randn(hidden_size))
        self.Vo = nn.Parameter(torch.randn(hidden_size, output_size))
        self.co = nn.Parameter(torch.randn(output_size))

    def forward(self, inputs, h0=None):
        """
        inputs: (seq_len, batch_size, input_size)
        h0: (batch_size, hidden_size)
        """

        seq_len, batch_size, _ = inputs.size()

        if h0 is None:
            ht = torch.zeros(batch_size, self.hidden_size, device = inputs.device)
        else:
            ht=h0
        
        outputs = []
        hidden_states = [ht]

        #equations from the provided image->forward pass only

        for t in range(seq_len):
            xt = inputs[t]
            at = self.bh + ht @ self.Wh + xt @ self.Ux 
            ht = torch.tanh(at)

            hidden_states.append(ht)
            ot = self.co + ht @ self.Vo 
            # ot = torch.softmax(ot, dim=1)
            # while the network uses softmax computation at this step
            # we remove it because it is calculated at the loss computation step anyway
            # redundant calculations may give numeric insntability
            outputs.append(ot)
        
        return torch.stack(outputs),ht

def compute_cross_entropy_loss(output, target):
    """
    output: (seq_len, vocab_size) we removed batch_size, remember?
    target: (seq_len)
    """
    total_loss = 0.0

    for t in range(target.size(0)):
        logits = output[t]
        target_index = target[t]

        zy = logits[target_index]
        log_sum_exp = torch.logsumexp(logits, dim=0)
        loss_t = -(zy - log_sum_exp)
        total_loss += loss_t

    return total_loss/target.size(0)

def preprocess(sentence):
    """
    sentence: input text to be learned from
    vocab: mapping of tokens->index of all unique tokens
    inv_vocab: inverse mapping index->tokens (as a sample decoder)
    target: true labels for the model
    words: all tokens in sentence
    """
    words = sentence.lower().split()

    vocab = {word : idx for idx, word in enumerate(set(words))}
    inv_vocab = {idx : word for word, idx in vocab.items()}
    vocab_size = len(vocab)

    seq_len = len(words)
    batch_size = 1

    x = torch.zeros(seq_len-1, batch_size, vocab_size)
    for t in range(seq_len-1):
        word = words[t]
        x[t, 0, vocab[word]] = 1
    
    target = torch.tensor([vocab[words[t+1]]for t in range(seq_len-1)])
    
    return x, vocab, inv_vocab, target, words

def train(num_epochs, rnn, optimizer, target, x):
    """
    num_epochs: no of training loops for the rnn
    rnn: object of the RNN class
    optimizer: object of torch.optim class here
    target: true labels
    x: input ids stacked as a tensor for the RNN class
    """
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output, _ = rnn(x)
        # output = output.view(-1, vocab_size)
        output = output.squeeze(1)
        
        # loss = loss_fn(output,target)
        loss = compute_cross_entropy_loss(output, target)
        loss.backward()
        optimizer.step()

        if (epoch+1)%10==0:
          print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

  
if __name__=="__main__":
    sentence = """The red-capped parrot (Purpureicephalus spurius) is a species of broad-tailed parrot native to southwest Western Australia. Described by Heinrich Kuhl in 1820, it is classified in its own genus owing to its distinctive elongated beak. Its closest relative is the mulga parrot. It is not easily confused with other parrot species; both adult sexes have a bright crimson crown, green-yellow cheeks, and a distinctive long bill. The wings, back, and long tail are dark green, and the underparts are purple-blue. Found in woodland and open savanna country, the red-capped parrot consumes seeds (particularly of eucalypts), flowers, berries, and occasionally insects. Nesting takes place in tree hollows. Although the red-capped parrot has been shot as a pest, and affected by land clearing, the population is growing and the species is not threatened."""

    x, vocab, inv_vocab, target, words = preprocess(sentence)
    input_size = len(vocab)
    hidden_size = 8
    output_size = len(vocab)
    num_epochs = 300
    batch_size = 1

    rnn = RNN(input_size, hidden_size, output_size)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)
    # loss_fn = nn.CrossEntropyLoss()

    print("Training the RNN...")
    train(num_epochs, rnn, optimizer, target,x)

    print('-'*30)
    #NEXT WORD#
    with torch.no_grad():
        output, _ = rnn(x)
        
        # Get predicted indices and words
        predicted_indices = torch.argmax(output.squeeze(1), dim=1)
        decoded_words = [inv_vocab[idx.item()] for idx in predicted_indices]
        
        # print("\nInput sequence:", words[:-1])
        # print("Target words:", words[1:])
        # print("Predicted words:", decoded_words)

    print('-'*30)
    #NEW SEQUENCE#
    print("\nGenerating additional words:")
    context = x  # Use the same context for generation
    current_hidden = None
    generated_sequence = words.copy()

    # Generate 5 more words
    num_words_to_generate = 5
    for _ in range(num_words_to_generate):
        with torch.no_grad():
            output, current_hidden = rnn(context, current_hidden)
            last_output = output[-1]
            next_word_idx = torch.argmax(last_output, dim=1).item()
            next_word = inv_vocab[next_word_idx]
            generated_sequence.append(next_word)
            
            # Update context for next prediction
            next_input = torch.zeros(1, batch_size, len(vocab))
            next_input[0, 0, next_word_idx] = 1
            context = next_input

    print('-'*30)
    print("Generated sequence:", " ".join(generated_sequence))