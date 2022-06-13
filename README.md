# excessive-gap-technique

Solving $ \min_{x \in \Delta_n} \max_{y \in \Delta_m} x'Ay $ by several algorithms 

## Features

- No-Regret Learning
  - Multiplicative Weights Update Algorithm (MWU)
  - Online Gradient Descent (OGD)
- Excessive gap technique
  - Gradient mapping (with Euclidean distance)
  - Bregman projection (with Entropy distance)

## Benchmark

![](https://user-images.githubusercontent.com/34413567/173423317-fa185fe8-e422-4a16-9f66-5e25d40e443c.png)

## References

- Zinkevich, Martin. "Online convex programming and generalized infinitesimal gradient ascent." Proceedings of the 20th international conference on machine learning (icml-03). 2003.
- Freund, Yoav, and Robert E. Schapire. "Adaptive game playing using multiplicative weights." Games and Economic Behavior 29.1-2 (1999): 79-103.
- Nesterov, Yu. "Smooth minimization of non-smooth functions." Mathematical programming 103.1 (2005): 127-152.
- Nesterov, Yu. "Excessive gap technique in nonsmooth convex minimization." SIAM Journal on Optimization 16.1 (2005): 235-249.
