# Agnostic Risk Parity: Taming Known and Unknown-Unknowns

**Paper**: Benichou, Lempérière, Sérié, Kockelkoren, Seager, Bouchaud & Potters (2016), arXiv:1610.08818 — `paper.pdf`.

## Status

Implemented as an algorithm experiment on the local factor zoo. The original paper's
110-futures panel is proprietary and is not claimed to be reproduced. The experiment
uses the paper's central ARP allocation, `C^(-1/2)p`, and compares it with an equal-risk
signal portfolio and raw/cleaned Markowitz allocations under identical volatility and
transaction-cost assumptions. Ledoit-Wolf cleaning is used instead of the paper's RIE.
