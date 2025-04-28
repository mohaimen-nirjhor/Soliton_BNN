# Soliton Solution using Bilinear Neural Network (BNN)

This project implements a **1-soliton solution** using a **Bilinear Neural Network (BNN)** approach with **Hirota's bilinear transformation**.  
It trains a neural network to learn the F-function that reconstructs the soliton profile and satisfies the bilinear form of the PDE.

---

## Project Features
- Models the 1-soliton solution with a neural network.
- Implicitly learns the F-function using Hirota’s transformation.
- Computes second derivatives for reconstructing the solution \( u(x,t) \).
- Minimizes a custom loss combining:
  - Mean squared error (MSE)
  - Error in reconstructed \( u(x,t) \)
  - Hirota bilinear residual
- Visualizes results with:
  - 1D soliton profile
  - 3D surface plot
  - Contour plot

---

## Requirements

- Python 3.x
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

You can install the dependencies with:

```bash
pip install torch numpy matplotlib
```

---

## How to Run

1. Make sure your environment has the required libraries.
2. Run the script:

```bash
python your_script_name.py
```

After training, the program automatically shows:
- A 1D soliton slice at fixed time
- A 3D surface plot of \( u(x,t) \)
- A contour plot of \( u(x,t) \)

---

## File Structure

| File | Description |
|:-----|:------------|
| `Soliton_BNN.py` | Main script for training and visualizing the soliton solution |

---

## Training Details

- **Hidden layer size**: 20 neurons
- **Optimizer**: Adam
- **Learning rate scheduler**: ReduceLROnPlateau
- **Epochs**: 1000
- **Special design choices**:
  - `Softplus` activation on output to ensure \( F(x,t) > 0 \)
  - Hirota bilinear residual computed and penalized during training

---

## Notes

- This model currently focuses on **single soliton** solutions.
- Can be extended to multiple solitons by modifying the F-function and training setup.
- Random wave number \( k \) is sampled from a uniform range [0.8, 1.2].

---

## License
This project is open source under the [MIT License](LICENSE).

