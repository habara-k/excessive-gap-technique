# excessive-gap-technique

Solving $ \min_{x \in \Delta_n} \max_{y \in \Delta_m} x'Ay $ by several algorithms 

## Features

- No-Regret Learning
  - Multiplicative Weights Update Algorithm (MWU)
  - Regret Matching (RM)
  - Online Gradient Descent (OGD)
- Excessive gap technique (EGT)
  - Euclidean distance (using Gradient mapping)
  - Entropy distance (using Bregman projection)

## Benchmark

```
python main.py --step 100000 --seed 0 -n 1000 -m 1000
```

![](https://user-images.githubusercontent.com/34413567/173562782-99ba0d78-c346-4492-91ad-c03eed14b9ec.png)


## References

- Freund, Yoav, and Robert E. Schapire. "Adaptive game playing using multiplicative weights." Games and Economic Behavior 29.1-2 (1999): 79-103.
- Hart, Sergiu, and Andreu Mas‐Colell. "A simple adaptive procedure leading to correlated equilibrium." Econometrica 68.5 (2000): 1127-1150.
- Zinkevich, Martin. "Online convex programming and generalized infinitesimal gradient ascent." Proceedings of the 20th international conference on machine learning (icml-03). 2003.
- Nesterov, Yu. "Smooth minimization of non-smooth functions." Mathematical programming 103.1 (2005): 127-152.
- Nesterov, Yu. "Excessive gap technique in nonsmooth convex minimization." SIAM Journal on Optimization 16.1 (2005): 235-249.
- Wang, Weiran, and Miguel A. Carreira-Perpinán. "Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application." arXiv preprint arXiv:1309.1541 (2013).
