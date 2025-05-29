# Scratch Lab

# X from Scratch

**X from Scratch** is a collection of from-scratch implementations of machine learning models, frameworks, and techniques using PyTorch (for now).

## Implementations

### Currently Available

1. **Vanilla RNN from Scratch**  
   - Implemented using PyTorch tensor operations
   - Includes step-by-step processing of sequence data and training loop
   - Demonstrates forward pass, hidden state updates, and manual loss computation
   - [Explanation](https://selective-jersey-d55.notion.site/RNN-from-scratch-1fa25c8ce8de80629048f528e84a6136?pvs=4)

2. **Backpropagation of a Random Network**  
   - Manually built feedforward network with custom backpropagation
   - Fully connected layers with ReLU and Tanh activations implemented via basic PyTorch ops
   - Gradients computed and applied without `autograd` for clarity
   - [Explanation](https://selective-jersey-d55.notion.site/How-To-Write-The-Backward-Pass-of-any-Network-20025c8ce8de802cb865ea357c9b2414?pvs=4)

3. **Byte Pair Encoding**  
   - Implemented the BPE algorithm as a prerequisite to Sentence Piece
   - That is it lol there isn't much to it
   - [Explanation](https://selective-jersey-d55.notion.site/Implementing-BPE-13f25c8ce8de807eb42bd85953092842?pvs=4)

4. **Decoding By Contrasting Layer (DoLa)**
   - Implemented DoLa's generation loop for auto-regressive generation for a random model
   - [OG Paper](https://arxiv.org/abs/2309.03883)

### 🧪 Planned Additions

- Sentence Piece
- Adam Optimizer built manually  
- Transformer encoder/decoder blocks  
- Graph Neural Networks  

## Getting Started

Clone the repo:

```bash
git clone https://github.com/yourusername/x-from-scratch.git
cd x-from-scratch



