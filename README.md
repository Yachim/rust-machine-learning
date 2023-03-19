# No Code Machine Learning

Suppose we have this sample network:

<p align="center">
  <img alt="Sample network" src="docs/net.png">
  
  <p align="center">
    <sup>$X$...inputs</sup>
</p>
</p>

<!-- definition of the activation -->
```math
a^{(l)}_j = f_l(z^{(l)}_j)
```

Where $f_l$ is the activation function for layer $l$, e.g., sigmoid:

<!-- example of activation with the sigmoid activation function -->
```math
a^{(l)}_j = \sigma(z^{(l)}_j)
```

<!-- definition of z -->
```math
z^{(l)}_j = \sum_{k = 0}^{n_{l - 1} - 1} (a^{(l - 1)}_k w^{(l)}_{jk}) + b^{(l)}_j
```
<p align="center">
  <sup>$n_l$...number of inputs in the layer $l$</sup>
</p>

<p align="center">
  <sup>$l$...any layer in the network</sup>
</p>

## All formulas for all a and z values

### $L = 0$
<!-- formula for a^{(0)}_0 -->
```math
a^{(0)}_0 = f_0(z^{(0)}_0)
```

<!-- formula for z^{(0)}_0 -->
```math
z^{(0)}_0 = {
  X_0 w^{(0)}_{00} + 
  X_1 w^{(0)}_{01} + 
  b^{(0)}_0
}
```

<!-- formula for a^{(0)}_1 -->
```math
a^{(0)}_1 = f_0(z^{(0)}_1)
```

<!-- formula for z^{(1)}_1 -->
```math
z^{(0)}_1 = {
  X_0 w^{(0)}_{10} + 
  X_1 w^{(0)}_{11} + 
  b^{(0)}_1
}
```

### $L = 1$
<!-- formula for a^{(1)}_0 -->
```math
a^{(1)}_0 = f_1(z^{(1)}_0)
```

<!-- formula for z^{(1)}_0 -->
```math
z^{(1)}_0 = {
  a^{(0)}_0 w^{(1)}_{00} + 
  a^{(0)}_1 w^{(1)}_{01} + 
  b^{(1)}_0
}
```

<!-- formula for a^{(3)}_1 -->
```math
a^{(1)}_1 = f_1(z^{(1)}_1)
```

<!-- formula for z^{(3)}_1 -->
```math
z^{(1)}_1 = {
  a^{(0)}_0 w^{(1)}_{10} + 
  a^{(0)}_1 w^{(1)}_{11} + 
  b^{(1)}_1
}
```

## Activation functions and their derivatives

The activation function is noted as $f_l$ (activation function for the layer $l$). The derivative is noted as $f'_{l}$.
 
### Sigmoid

<!-- definition of sigmoid activation function -->
```math
\sigma(z_j) = \frac{1}{1 + e^{-z_j}}
```

<!-- derivative of sigmoid -->
```math
\sigma'(z_j) = \sigma(z_j)(1 - \sigma(z_j))
```

### ReLU

<!-- definition of relu activation function -->
```math
ReLU(z_j) = max(0, z_j)
```

<!-- derivative of relu -->
```math
ReLU'(z_j) = \left\{
  \begin{array}{ l l }
    1 \qquad \textrm{if $z_j > 0$} \\ 0 \qquad \textrm{if $z_j \leq 0$*}
  \end{array}
\right.
```

<p align="center">
  <sup>*The derivative at zero is not defined. The predefined function in the program returns 0.</sup>
</p>

### Softmax

The code's implementation normalizes inputs.

<!-- definition of softmax -->
```math
Softmax(z_j) = \frac{e^{z_j - max_z}}{sum}
```

<!-- because github had problem processing sum in denominator -->
```math
sum = \sum_{i=0}^{n_z - 1} e^{z - max_z}
```

<p align="center">
  <sup>$max_z$...maximum value of all values in the layer</sup>
</p>

<p align="center">
  <sup>$n_z$...number of values in the layer</sup>
</p>

<!-- derivative of softmax -->
```math
\frac{\partial Softmax(z_i)}{\partial z_j} = Softmax(z_i) (\delta_{ij} - Softmax(z_j))
```

<!-- the value of x -->
```math
\delta_{ij} = \left\{
  \begin{array}{ l l }
    1 \qquad \textrm{if $i = j$} \\ 0 \qquad \textrm{if $i \neq j$}
  \end{array}
\right.
```

## Derivatives of cost functions

<!-- partial derivative of C with respect to w^{(L)}_{jk} -->
```math
\frac{\partial C}{\partial w^{(L)}_{jk}} = {
  \frac{\partial C}{\partial a^{(L)}_j}
  \frac{\partial a^{(L)}_j}{\partial z^{(L)}_j}
  \frac{\partial z^{(L)}_j}{\partial w^{(L)}_{jk}}
}
```

<!-- partial derivative of C with respect to b^{(L)}_j -->
```math
\frac{\partial C}{\partial b^{(L)}_j} = {
  \frac{\partial C}{\partial a^{(L)}_j}
  \frac{\partial a^{(L)}_j}{\partial z^{(L)}_j}
  \frac{\partial z^{(L)}_j}{\partial b^{(L)}_j}
}
```

<!-- partial derivative of C with respect to a^{(L - 1)}_k -->
```math
\frac{\partial C}{\partial a^{(L - 1)}_k} = {
  \sum_{j=0}^{n_L - 1}
  \frac{\partial C}{\partial a^{(L)}_j}
  \frac{\partial a^{(L)}_j}{\partial z^{(L)}_j}
  \frac{\partial z^{(L)}_j}{\partial a^{(L - 1)}_k}
}
```

<p align="center">
  <sup>$L$...last layer of the network</sup>
</p>

The common part for all three equations:

```math
root^{(L)}_j = 
  \frac{\partial C}{\partial a^{(L)}_j}
  \frac{\partial a^{(L)}_j}{\partial z^{(L)}_j}
```

After substituting $root$:

<!-- partial derivative of C with respect to w^{(L)}_{jk} (with root) -->
```math
\frac{\partial C}{\partial w^{(L)}_{jk}} = {
  root^{(L)}_j
  \frac{\partial z^{(L)}_j}{\partial w^{(L)}_{jk}}
}
```

<!-- partial derivative of C with respect to b^{(L)}_j (with root) -->
```math
\frac{\partial C}{\partial b^{(L)}_j} = {
  root^{(L)}_j
  \frac{\partial z^{(L)}_j}{\partial b^{(L)}_j}
}
```

<!-- partial derivative of C with respect to a^{(L - 1)}_k (with root) -->
```math
\frac{\partial C}{\partial a^{(L - 1)}_k} = {
  \sum_{j=0}^{n_L - 1}
  root^{(L)}_j
  \frac{\partial z^{(L)}_j}{\partial a^{(L - 1)}_k}
}
```

<br>

<!-- partial derivative of z^{(L)}_j with respect to w^{(L)}_{jk} -->
```math
\frac{\partial z^{(L)}_j}{\partial w^{(L)}_{jk}} =
  a^{(L - 1)}_k
```

<!-- partial derivative of z^{(L)}_j with respect to b^{(L)}_j -->
```math
\frac{\partial z^{(L)}_j}{\partial b^{(L)}_j} = 
  1
```
<!-- partial derivative of z^{(L)}_j with respect to a^{(L - 1)}_k -->
```math
\frac{\partial z^{(L)}_j}{\partial a^{(L - 1)}_k} =
  w^{(L)}_{jk}
```

For any layer $l$ in the network:

<!-- partial derivative of C with respect to w^{(l)}_{jk} -->
```math
\frac{\partial C}{\partial w^{(l)}_{jk}} = {
  \frac{\partial C}{\partial a^{(l)}_j}
  \frac{\partial a^{(l)}_j}{\partial z^{(l)}_j}
  a^{(l - 1)}_k
} = {
  root^{(l)}_j
  a^{(l - 1)}_k
}
```

<!-- partial derivative of C with respect to b^{(l)}_j -->
```math
\frac{\partial C}{\partial b^{(l)}_j} = {
  \frac{\partial C}{\partial a^{(l)}_j}
  \frac{\partial a^{(l)}_j}{\partial z^{(l)}_j}
} =
  root^{(l)}_j
```

Where:

<!-- partial derivative of C with respect to a^{(l)}_k if l = L -->
```math
\frac{\partial C}{\partial a^{(l)}_j} = \frac{\partial C}{\partial a^{(L)}_j} \qquad 
\textrm{if $l=L$}
```

<!-- partial derivative of C with respect to a^{(l)}_k if l != L -->
```math
\frac{\partial C}{\partial a^{(l)}_j} = {
  \sum_{j=0}^{n_{l + 1} - 1}
  \frac{\partial C}{\partial a^{(l + 1)}_j}
  \frac{\partial a^{(l + 1)}_j}{\partial z^{(l + 1)}_j}
  w^{(l + 1)}_{jk}
} = {
  \sum_{j=0}^{n_{l + 1} - 1}
  root^{(l + 1)}_j
  w^{(l + 1)}_{jk}
} \qquad
\textrm{otherwise}
```

### Mean Squared Error (MSE)

<!-- the definition of mse -->
```math
C = MSE = {
  \frac{1}{n_L}
  \sum_{j = 0}^{n_L-1}(a^{(L)}_j - y_j)^2
}
```

<p align="center">
  <sup>$y$...the expected values</sup>  
</p>

<!-- the derivative of mse -->
```math
\frac{\partial C}{\partial a^{(L)}_j} = 2(a^{(L)}_j - y_j)
```
