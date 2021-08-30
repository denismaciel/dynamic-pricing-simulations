# Dynamic Pricing Simulations

Simulation-based comparison between dynamic pricing algorithms

## How to render notebooks

Make sure all relevant notebooks are listed in `PATH` in`scripts/compile_notebooks.py`.

Run:

```
FIGS_DIR=/home/denis/Personal/MasterThesis/dynamic-pricing-simulations/figs ./venv/bin/python3 scripts/compile_notebooks.py
```
This will sync `.py` with `.ipynb` and render `.ipynb` as `.md`. All notebooks
will have their content executed. To change that (why would you? Doesn't hurt to
wait a little), toggle the `execute` variable in `main` function by hand.



% This work is divided into sections. In the section
% ``\nameref{sec:limited_inventory}'', we explore the scenario where a firm is not
% able to reorder products when initial stock is sold out. By assuming the demand
% function is fully known and non-stochastic, we discuss what is the optimal way
% to price the product during the sales season. Next, in the section
% ``\nameref{Sec:DemandAndLearning}'', we try to answer the questions: How should
% a firm balance the need of learning the demand while at the same time optimizing
% prices for revenue maximization? Should it explore different price points to
% learn or should it set the price believed to be optimal? We discuss different
% strategies a firm might adopt while setting prices and learning the demand. The
% section ``\nameref{sec:learning_demand_under_limited_inventory}'' follows.
% There, the combination of limited inventory and demand learning is discussed. We
% implement the Thompson-sampling-based agents (TS-Fixed and TS-Update) described
% in \cite{ferreira2018online}. We manage to replicate their numerical results in
% the section ``\nameref{sec:replication}''. Finally, in the section
% ``\nameref{sec:competition}'', we simulate the agents side by side in a market.
% That way, the demand for an agent is affected by the price set by another agent.
% The simulation results show that the edge that Thompson-sampling agents have in
% a monopolistic scenario can be reduced significantly once a competitor is
% present in the market.

To replicate the plots from Section \nameref{sec:limited_inventory}, run
\texttt{simulations/deterministic\_models/sim.py}.
\nameref{Sec:DemandAndLearning}
\nameref{sec:learning_demand_under_limited_inventory}
\nameref{sec:replication}
\nameref{sec:competition}

Figure \ref{fig:bayesian_updating2} simulations/exploration_exploitation/bayesian_updating.R  

\begin{itemize}
    \item Figure \ref{fig:deterministic_revenue} and Table
        \ref{tab:10_period_demands}, 
    \item Figure \ref{fig:deterministic_revenue} and Table \ref{tab:10_period_demands}
\end{itemize}

% deterministic_models
% exploration_exploitation
% online_network_revenue_management
% replication_online_net_rev_mgmt
% demand
% competition
