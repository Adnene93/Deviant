python ../../../Deviant-Code/main.py quality_YELP_PVALUE.json -p --nb_random_sample 1000 --nb_items_individuals 0.25 --nb_items_entities 0.25 --nb_attributes_individuals 3 --nb_attributes_entities 2  --threshold_objects 0.001 --threshold_individuals 0.01 --significance 0.05 --using_approxmations --results_destination yelp_nb_entities  --first_run  ;
python ../../../Deviant-Code/main.py quality_YELP_PVALUE.json -p --nb_random_sample 1000 --nb_items_individuals 0.25 --nb_items_entities 0.5 --nb_attributes_individuals 3 --nb_attributes_entities 2  --threshold_objects 0.001 --threshold_individuals 0.01 --significance 0.05 --using_approxmations --results_destination yelp_nb_entities ;
python ../../../Deviant-Code/main.py quality_YELP_PVALUE.json -p --nb_random_sample 1000 --nb_items_individuals 0.25 --nb_items_entities 0.75 --nb_attributes_individuals 3 --nb_attributes_entities 2  --threshold_objects 0.001 --threshold_individuals 0.01 --significance 0.05 --using_approxmations --results_destination yelp_nb_entities ;
python ../../../Deviant-Code/main.py quality_YELP_PVALUE.json -p --nb_random_sample 1000 --nb_items_individuals 0.25 --nb_items_entities 1 --nb_attributes_individuals 3 --nb_attributes_entities 2  --threshold_objects 0.001 --threshold_individuals 0.01 --significance 0.05 --using_approxmations --results_destination yelp_nb_entities ;

python add_column_csv.py yelp_nb_entities_performance_test.csv ;


python ../../../Deviant-Code/main.py quality_YELP_PVALUE.json -p --nb_random_sample 1000 --nb_items_individuals 0 --nb_items_entities 1 --nb_attributes_individuals 0 --nb_attributes_entities 2  --threshold_objects 0.001 --threshold_individuals 0.01 --significance 0.05 --using_approxmations --results_destination yelp_nb_individuals  --first_run ;
python ../../../Deviant-Code/main.py quality_YELP_PVALUE.json -p --nb_random_sample 1000 --nb_items_individuals 0.25 --nb_items_entities 1 --nb_attributes_individuals 3 --nb_attributes_entities 2  --threshold_objects 0.001 --threshold_individuals 0.01 --significance 0.05 --using_approxmations --results_destination yelp_nb_individuals  ;
python ../../../Deviant-Code/main.py quality_YELP_PVALUE.json -p --nb_random_sample 1000 --nb_items_individuals 0.5 --nb_items_entities 1 --nb_attributes_individuals 3 --nb_attributes_entities 2  --threshold_objects 0.001 --threshold_individuals 0.01 --significance 0.05 --using_approxmations --results_destination yelp_nb_individuals ;
python ../../../Deviant-Code/main.py quality_YELP_PVALUE.json -p --nb_random_sample 1000 --nb_items_individuals 1 --nb_items_entities 1 --nb_attributes_individuals 3 --nb_attributes_entities 2  --threshold_objects 0.001 --threshold_individuals 0.01 --significance 0.05 --using_approxmations --results_destination yelp_nb_individuals ;

python add_column_csv.py yelp_nb_individuals_performance_test.csv ;


python ../../../Deviant-Code/main.py quality_YELP_PVALUE.json -p --nb_random_sample 1000 --nb_items_individuals 0.25 --nb_items_entities 1 --nb_attributes_individuals 3 --nb_attributes_entities 2  --threshold_objects 0.001 --threshold_individuals 0.01 --significance 0.05 --using_approxmations --results_destination yelp_critical  --first_run  ;
python ../../../Deviant-Code/main.py quality_YELP_PVALUE.json -p --nb_random_sample 1000 --nb_items_individuals 0.25 --nb_items_entities 1 --nb_attributes_individuals 3 --nb_attributes_entities 2  --threshold_objects 0.001 --threshold_individuals 0.01 --significance 0.01 --using_approxmations --results_destination yelp_critical  ;
python ../../../Deviant-Code/main.py quality_YELP_PVALUE.json -p --nb_random_sample 1000 --nb_items_individuals 0.25 --nb_items_entities 1 --nb_attributes_individuals 3 --nb_attributes_entities 2  --threshold_objects 0.001 --threshold_individuals 0.01 --significance 0.001 --using_approxmations --results_destination yelp_critical ;

python add_column_csv.py yelp_critical_performance_test.csv ;


python ../../../Deviant-Code/main.py quality_YELP_PVALUE.json -p --nb_random_sample 1000 --nb_items_individuals 0.25 --nb_items_entities 1 --nb_attributes_individuals 3 --nb_attributes_entities 2  --threshold_objects 0.05 --threshold_individuals 0.01 --significance 0.05 --using_approxmations --results_destination yelp_threshold_object  --first_run  ;
python ../../../Deviant-Code/main.py quality_YELP_PVALUE.json -p --nb_random_sample 1000 --nb_items_individuals 0.25 --nb_items_entities 1 --nb_attributes_individuals 3 --nb_attributes_entities 2  --threshold_objects 0.01 --threshold_individuals 0.01 --significance 0.05 --using_approxmations --results_destination yelp_threshold_object ;
python ../../../Deviant-Code/main.py quality_YELP_PVALUE.json -p --nb_random_sample 1000 --nb_items_individuals 0.25 --nb_items_entities 1 --nb_attributes_individuals 3 --nb_attributes_entities 2  --threshold_objects 0.001 --threshold_individuals 0.01 --significance 0.05 --using_approxmations --results_destination yelp_threshold_object  ;

python add_column_csv.py yelp_threshold_object_performance_test.csv ;

python ../../../Deviant-Code/main.py quality_YELP_PVALUE.json -p --nb_random_sample 1000 --nb_items_individuals 0.25 --nb_items_entities 0.25 --nb_attributes_individuals 3 --nb_attributes_entities 2  --threshold_objects 0.001 --threshold_individuals 0.01 --significance 0.05 --using_approxmations --compute_bootstrap_ci --results_destination yelp_nb_entities_BS  --first_run  ;
python ../../../Deviant-Code/main.py quality_YELP_PVALUE.json -p --nb_random_sample 1000 --nb_items_individuals 0.25 --nb_items_entities 0.5 --nb_attributes_individuals 3 --nb_attributes_entities 2  --threshold_objects 0.001 --threshold_individuals 0.01 --significance 0.05 --using_approxmations --compute_bootstrap_ci --results_destination yelp_nb_entities_BS ;
python ../../../Deviant-Code/main.py quality_YELP_PVALUE.json -p --nb_random_sample 1000 --nb_items_individuals 0.25 --nb_items_entities 0.75 --nb_attributes_individuals 3 --nb_attributes_entities 2  --threshold_objects 0.001 --threshold_individuals 0.01 --significance 0.05 --using_approxmations --compute_bootstrap_ci --results_destination yelp_nb_entities_BS ;
python ../../../Deviant-Code/main.py quality_YELP_PVALUE.json -p --nb_random_sample 1000 --nb_items_individuals 0.25 --nb_items_entities 1 --nb_attributes_individuals 3 --nb_attributes_entities 2  --threshold_objects 0.001 --threshold_individuals 0.01 --significance 0.05 --using_approxmations --compute_bootstrap_ci --results_destination yelp_nb_entities_BS ;

python add_column_csv.py yelp_nb_entities_BS_performance_test.csv ;

python ../../../Deviant-Code/main.py quality_YELP_PVALUE.json -p --nb_random_sample 1000 --nb_items_individuals 0 --nb_items_entities 1 --nb_attributes_individuals 0 --nb_attributes_entities 2  --threshold_objects 0.001 --threshold_individuals 0.01 --significance 0.05 --using_approxmations --compute_bootstrap_ci --results_destination yelp_nb_individuals_BS  --first_run ;
python ../../../Deviant-Code/main.py quality_YELP_PVALUE.json -p --nb_random_sample 1000 --nb_items_individuals 0.25 --nb_items_entities 1 --nb_attributes_individuals 3 --nb_attributes_entities 2  --threshold_objects 0.001 --threshold_individuals 0.01 --significance 0.05 --using_approxmations --compute_bootstrap_ci --results_destination yelp_nb_individuals_BS  ;
python ../../../Deviant-Code/main.py quality_YELP_PVALUE.json -p --nb_random_sample 1000 --nb_items_individuals 0.5 --nb_items_entities 1 --nb_attributes_individuals 3 --nb_attributes_entities 2  --threshold_objects 0.001 --threshold_individuals 0.01 --significance 0.05 --using_approxmations --compute_bootstrap_ci --results_destination yelp_nb_individuals_BS ;
python ../../../Deviant-Code/main.py quality_YELP_PVALUE.json -p --nb_random_sample 1000 --nb_items_individuals 1 --nb_items_entities 1 --nb_attributes_individuals 3 --nb_attributes_entities 2  --threshold_objects 0.001 --threshold_individuals 0.01 --significance 0.05 --using_approxmations --compute_bootstrap_ci --results_destination yelp_nb_individuals_BS ;

python add_column_csv.py yelp_nb_individuals_BS_performance_test.csv ;

python ../../../Deviant-Code/main.py quality_YELP_PVALUE.json -p --nb_random_sample 1000 --nb_items_individuals 0.25 --nb_items_entities 1 --nb_attributes_individuals 3 --nb_attributes_entities 2  --threshold_objects 0.001 --threshold_individuals 0.01 --significance 0.05 --using_approxmations --compute_bootstrap_ci --results_destination yelp_critical_BS  --first_run  ;
python ../../../Deviant-Code/main.py quality_YELP_PVALUE.json -p --nb_random_sample 1000 --nb_items_individuals 0.25 --nb_items_entities 1 --nb_attributes_individuals 3 --nb_attributes_entities 2  --threshold_objects 0.001 --threshold_individuals 0.01 --significance 0.01 --using_approxmations --compute_bootstrap_ci --results_destination yelp_critical_BS  ;
python ../../../Deviant-Code/main.py quality_YELP_PVALUE.json -p --nb_random_sample 1000 --nb_items_individuals 0.25 --nb_items_entities 1 --nb_attributes_individuals 3 --nb_attributes_entities 2  --threshold_objects 0.001 --threshold_individuals 0.01 --significance 0.001 --using_approxmations --compute_bootstrap_ci --results_destination yelp_critical_BS ;

python add_column_csv.py yelp_critical_BS_performance_test.csv ;

python ../../../Deviant-Code/main.py quality_YELP_PVALUE.json -p --nb_random_sample 1000 --nb_items_individuals 0.25 --nb_items_entities 1 --nb_attributes_individuals 3 --nb_attributes_entities 2  --threshold_objects 0.05 --threshold_individuals 0.01 --significance 0.05 --using_approxmations --compute_bootstrap_ci --results_destination yelp_threshold_object_BS  --first_run  ;
python ../../../Deviant-Code/main.py quality_YELP_PVALUE.json -p --nb_random_sample 1000 --nb_items_individuals 0.25 --nb_items_entities 1 --nb_attributes_individuals 3 --nb_attributes_entities 2  --threshold_objects 0.01 --threshold_individuals 0.01 --significance 0.05 --using_approxmations --compute_bootstrap_ci --results_destination yelp_threshold_object_BS ;
python ../../../Deviant-Code/main.py quality_YELP_PVALUE.json -p --nb_random_sample 1000 --nb_items_individuals 0.25 --nb_items_entities 1 --nb_attributes_individuals 3 --nb_attributes_entities 2  --threshold_objects 0.001 --threshold_individuals 0.01 --significance 0.05 --using_approxmations --compute_bootstrap_ci --results_destination yelp_threshold_object_BS  ;

python add_column_csv.py yelp_threshold_object_BS_performance_test.csv ;