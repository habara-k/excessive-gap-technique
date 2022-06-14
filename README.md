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

```
python main.py --step 10000 --seed 0 -n 1000 -m 1000
```
![](https://user-images.githubusercontent.com/34413567/173515842-a135744f-3e94-4bc4-81f4-2fa0465c56b6.png)



## References

- Zinkevich, Martin. "Online convex programming and generalized infinitesimal gradient ascent." Proceedings of the 20th international conference on machine learning (icml-03). 2003.
- Freund, Yoav, and Robert E. Schapire. "Adaptive game playing using multiplicative weights." Games and Economic Behavior 29.1-2 (1999): 79-103.
- Nesterov, Yu. "Smooth minimization of non-smooth functions." Mathematical programming 103.1 (2005): 127-152.
- Nesterov, Yu. "Excessive gap technique in nonsmooth convex minimization." SIAM Journal on Optimization 16.1 (2005): 235-249.
- Wang, Weiran, and Miguel A. Carreira-Perpinán. "Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application." arXiv preprint arXiv:1309.1541 (2013).
