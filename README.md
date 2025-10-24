# python-ed_sim_simple.py

Install required packages:

pip install pandas matplotlib

Save the script above as ed_sim.py and run:

python ed_sim.py

Output folder: ed_sim_outputs/ (contains PNG figures and ed_simulation_summary.csv and ed_simulation_report.md).

To modify scenarios:

Edit the scenarios dictionary in main() (adjust n_servers, arrival_mean, service_mean).

Re-run.

To extend the model:

Replace exponential service distribution with random.lognormvariate or a gamma; add triage priority queues; add time-of-day arrival rate function.
