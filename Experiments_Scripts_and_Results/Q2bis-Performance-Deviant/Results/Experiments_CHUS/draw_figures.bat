python ../../../../Deviant-Code/main.py chus_nb_entities_performance_test.csv -f --x_axis_attribute nb_contexts_attributes_items --fig_algo DEPICT --linear_scale_bars
python ../../../../Deviant-Code/main.py chus_nb_individuals_performance_test.csv -f --x_axis_attribute nb_individuals_attributes_items --fig_algo DEPICT --linear_scale_bars
python ../../../../Deviant-Code/main.py chus_threshold_object_performance_test.csv -f --x_axis_attribute support_context_threshold --fig_algo DEPICT --linear_scale_bars
python ../../../../Deviant-Code/main.py chus_critical_performance_test.csv -f --x_axis_attribute quality_threshold --fig_algo DEPICT --linear_scale_bars 

python ../../../../Deviant-Code/main.py chus_nb_entities_BS_performance_test.csv -f --x_axis_attribute nb_contexts_attributes_items --fig_algo DEPICT_BS --linear_scale_bars
python ../../../../Deviant-Code/main.py chus_nb_individuals_BS_performance_test.csv -f --x_axis_attribute nb_individuals_attributes_items --fig_algo DEPICT_BS --linear_scale_bars
python ../../../../Deviant-Code/main.py chus_threshold_object_BS_performance_test.csv -f --x_axis_attribute support_context_threshold --fig_algo DEPICT_BS --linear_scale_bars
python ../../../../Deviant-Code/main.py chus_critical_BS_performance_test.csv -f --x_axis_attribute quality_threshold --fig_algo DEPICT_BS --linear_scale_bars 
pause