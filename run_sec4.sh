results_path=$(PYTHONPATH=. python scripts/sec4/nonmyopic_simulations.py --config scripts/sec4/figure_4_and_EC1.json)
PYTHONPATH=. python scripts/sec4/nonmyopic_plots.py --config scripts/sec4/figure_4_and_EC1.json --results $results_path
echo
results_path=$(PYTHONPATH=. python scripts/sec4/nonmyopic_simulations.py --config scripts/sec4/figure_EC2.json)
PYTHONPATH=. python scripts/sec4/nonmyopic_plots.py --config scripts/sec4/figure_EC2.json --results $results_path
echo
results_path=$(PYTHONPATH=. python scripts/sec4/nonmyopic_simulations.py --config scripts/sec4/figure_EC3.json)
PYTHONPATH=. python scripts/sec4/nonmyopic_plots.py --config scripts/sec4/figure_EC3.json --results $results_path
echo
results_path=$(PYTHONPATH=. python scripts/sec4/nonmyopic_simulations.py --config scripts/sec4/figure_EC4.json)
PYTHONPATH=. python scripts/sec4/nonmyopic_plots.py --config scripts/sec4/figure_EC4.json --results $results_path