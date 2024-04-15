In word embeddings, we have the following optimization problem

$
max log p(x; rho, alpha) - lambda norm(rho)_F^2 - lambda norm(alpha)_F^2
$

We know that the likelihood $p(x; rho, alpha)$ is invariant wrt. invertable linear transformations $W$

$
log p(x; rho, alpha) = log p(x; W rho, W^(-1) alpha)
$

For each $rho, alpha$, there is then a rotation $W$ that optimizes the loss given the value of the likelihood $p(x; rho, alpha)$, minimizing the following quantity

$
lambda norm(W rho)_F^2 + lambda norm(W^(-1) alpha)_F^2 prop & norm(W rho)_F^2 + norm(W^(-1) alpha)_F^2 \
=  & "Tr"(rho^T W^T W rho) + "Tr"(alpha^T W^(-1 T) W^(-1) alpha) \
=  & "Tr"(rho^T W^T W rho) + "Tr"(alpha^T (W W^T)^(-1) alpha) \
$

Differentiating this, we obtain#footnote[Derived using the Matrix Cookbook and matrixcalculus.org. Verified via TensorFlow AutoDiff.]

$
diff / (diff W) "Tr"(rho^T W^T W rho) &= 2 W rho rho^T\
diff / (diff W) "Tr"(alpha^T (W W^T)^(-1) alpha) &= - 2 W^(-T) W^(-1) alpha alpha^T W^(-T) \
$

Finding an analytical solution for the optimal rotation for $W$ is complicated. It can be found pretty easily using gradient descent, though.

As #cite(<mu2019revisiting>, form: "prose") show, using a spherical prior identifies the posterior up until an orthogonal transformation ($W^T W= I$). For this reason, if we simulate $rho$ and $alpha$, the spherical prior is going to not only find $W rho$ and $W^(-1) alpha$ with any transformation $W$, but the optimal transformation. That might yield different $rho$ and $alpha$ than the simulated ones.

To counter this, we want to simulate $rho$ and $alpha$, and then find the $W$ that minimizes the Frobenius norm of these. That would then be the reference embedding that we compare the values with. Note that the conditional probabilities $sigma(alpha_w^T rho_v)$ remain unchanged regardless, while cosine similarities between the words ($"cossim"(rho_v, rho_w)$) and vector norms ($norm(rho_v)$, $norm(alpha_w)$) will change after making this adjustment.

#bibliography("references.bib", style: "american-psychological-association")