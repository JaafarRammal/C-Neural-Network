// neural_net.cpp

#include <iostream>
#include <vector>
#include <cstdlib>
#include <assert.h>
#include <math.h>
#include <fstream>
#include <sstream>

// ******************** class training_data ********************

class training_data{
public:
	training_data(const std::string filename);
	bool is_eof(void){
		return training_data_file.eof();
	}
	void get_topology(std::vector<unsigned> &topology);

	// Returns the number of input values read from the file:
	unsigned get_next_inputs(std::vector<double> &input_values);
	unsigned get_target_outputs(std::vector<double> &target_output_values);

private:
	std::ifstream training_data_file;
};

void training_data::get_topology(std::vector<unsigned> &topology){
	std::string line;
	std::string label;

	getline(training_data_file, line);
	std::stringstream ss(line);
	ss >> label;
	if(this->is_eof() || label.compare("topology:") != 0){
		abort();
	}

	while(!ss.eof()){
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}
	return;
}

training_data::training_data(const std::string filename){
	training_data_file.open(filename.c_str());
}


unsigned training_data::get_next_inputs(std::vector<double> &input_values){
    input_values.clear();

    std::string line;
    getline(training_data_file, line);
    std::stringstream ss(line);

    std::string label;
    ss >> label;
    if (label.compare("in:") == 0){
        double oneValue;
        while (ss >> oneValue){
            input_values.push_back(oneValue);
        }
    }

    return input_values.size();
}

unsigned training_data::get_target_outputs(std::vector<double> &target_output_values){
    target_output_values.clear();

    std::string line;
    getline(training_data_file, line);
    std::stringstream ss(line);

    std::string label;
    ss>> label;
    if (label.compare("out:") == 0){
        double oneValue;
        while (ss >> oneValue){
            target_output_values.push_back(oneValue);
        }
    }

    return target_output_values.size();
}

class neuron;
typedef std::vector<neuron> layer;

// ******************** struct connection ********************

struct connection{
	
	double weight;
	double delta_weight;

};

// ******************** class neuron ********************

class neuron{

public:
	neuron(unsigned num_outputs, unsigned in_neuron_index);
	void set_output_value(double input_value) {neurone_output_value = input_value;}
	double get_output_value(void) const {return neurone_output_value;}
	void feed_forward(const layer &previous_layer);
	void calculate_output_gradients(double target_value);
	void calculate_hidden_gradients(const layer &next_layer);
	void update_input_wieight(layer &previous_layer);

private:
	static double transfer_function(double x);
	static double transfer_function_derivative(double x);
	static double random_weigth(void){ return rand() / double(RAND_MAX);}
	double sum_dow(const layer &next_layer);
	double neurone_output_value;
	double neuron_gradient;
	static double eta; // 0.0: slow learner | 0.2: medium learner | 1.0: reckless learner | range [0.0..1.0]
	static double alpha; // 0.0: no momentum | 0.5: moderate momentum | range [0.0..n]
	unsigned neuron_index;
	std::vector<connection> neuron_output_weights;
};

double neuron::eta = 0.15;
double neuron::alpha = 0.5;


void neuron::update_input_wieight(layer &previous_layer){
	// update previous layer weights
	for(unsigned n = 0; n < previous_layer.size(); n++){
		neuron &current_neuron = previous_layer[n];
		double old_delta_weight = current_neuron.neuron_output_weights[neuron_index].delta_weight;
		double new_delta_weight = 
			// individual input magnified by the gradient and training rate
			eta
			* current_neuron.get_output_value()
			* neuron_gradient
			// also add momentum to the previous delta weight
			+ alpha
			* old_delta_weight;

		current_neuron.neuron_output_weights[neuron_index].delta_weight = new_delta_weight;
		current_neuron.neuron_output_weights[neuron_index].weight += new_delta_weight;
	}
}

double neuron::sum_dow(const layer &next_layer){
	double sum = 0.0;

	for(unsigned n = 0; n < next_layer.size() - 1; n++){
		sum += neuron_output_weights[n].weight * next_layer[n].neuron_gradient;
	}

	return sum;
}

void neuron::calculate_hidden_gradients(const layer &next_layer){
	double dow = sum_dow(next_layer);
	neuron_gradient = dow * neuron::transfer_function_derivative(neurone_output_value);
}

void neuron::calculate_output_gradients(double target_value){
	double delta = target_value - neurone_output_value;
	neuron_gradient = delta * neuron::transfer_function_derivative(neurone_output_value);
}

double neuron::transfer_function(double x){

	// will use tanh(x)
	return tanh(x);

}

double neuron::transfer_function_derivative(double x){

	// will use d(tanh(x))/dx
	return 1.0 - (x * x);

}

void neuron::feed_forward(const layer &previous_layer){
	double sum = 0.0;

	// sum the previous layer's outputs (which are our inputs)
	// include the bias node from the previous layer

	for(unsigned n = 0; n < previous_layer.size(); n++){
		sum += previous_layer[n].get_output_value() * previous_layer[n].neuron_output_weights[neuron_index].weight;
	}

	neurone_output_value = transfer_function(sum);
}

neuron::neuron(unsigned num_outputs, unsigned in_neuron_index){

	for(unsigned connection_index = 0; connection_index < num_outputs; connection_index++){
		neuron_output_weights.push_back(connection());
		neuron_output_weights.back().weight = random_weigth();
	}

	neuron_index = in_neuron_index;

}

// ******************** class net ********************

class net{

public:

	net(const std::vector<unsigned> &net_topology);

	// pass by const reference the inputs since they are not changed
	void feed_forward(const std::vector<double> &input_values);

	// pass by const reference the target outputs since they are not changed
	void back_propagate(const std::vector<double> &target_values);

	// pass by reference the outputs
	void get_results(std::vector<double> &results_values) const;


	double get_recent_average_error(void) const { return net_recent_average_error; }

private:

	double net_error;
	double net_recent_average_error;
	double net_recent_average_smoothing_factor;
	// net_layers[layer_index][neuron_index]
	std::vector<layer> net_layers;
};

void net::back_propagate(const std::vector<double> &target_values){

	// calculate overall net error (will use RMS of output errors)
	layer &output_layer = net_layers.back();
	net_error = 0.0;

	for(unsigned n = 0;  n < output_layer.size() - 1; n++){
		double delta = target_values[n] - output_layer[n].get_output_value();
		net_error += delta * delta;
	}

	net_error /= output_layer.size() - 1;
	net_error = sqrt(net_error); // RMS

	// implement a recent average measurment
	net_recent_average_error =
		(net_recent_average_error * net_recent_average_smoothing_factor + net_error)
		/ (net_recent_average_smoothing_factor + 1.0);

	// calculate output layer gradients
	for(unsigned n = 0; n < output_layer.size() - 1; n++){
		output_layer[n].calculate_output_gradients(target_values[n]);
	}

	// calculate gradients on hidden layers
	for(unsigned layer_index = net_layers.size() - 2; layer_index > 0; layer_index--){
		layer &hidden_layer = net_layers[layer_index];
		layer &next_layer = net_layers[layer_index + 1];

		for(unsigned n = 0; n < hidden_layer.size(); n++){
			hidden_layer[n].calculate_hidden_gradients(next_layer);
		}

	}

	// for all layers from output to first hidden layer,
	// update connection weights

	for(unsigned layer_index = net_layers.size() - 1; layer_index > 0; layer_index--){
		layer &current_layer = net_layers[layer_index];
		layer &previous_layer = net_layers[layer_index - 1];

		for(unsigned n = 0; n < current_layer.size(); n++){
			current_layer[n].update_input_wieight(previous_layer);
		}
	}

}

void net::feed_forward(const std::vector<double> &input_values){

	// assert inputs equal to size of first layer without the bias neuron
	assert(input_values.size() == net_layers[0].size() - 1);

	// assign the input values to the input neurons (index 0)
	for(unsigned i = 0; i < input_values.size(); i++){
		net_layers[0][i].set_output_value(input_values[i]);
	}

	// forward propagate (index 1 to n)
	for(unsigned layer_index = 1; layer_index < net_layers.size(); layer_index++){
		layer &previous_layer = net_layers[layer_index - 1];
		for(unsigned n = 0; n < net_layers[layer_index].size() - 1; n++){
			net_layers[layer_index][n].feed_forward(previous_layer); 
		}
	}

}

void net::get_results(std::vector<double> &results_values) const{

	results_values.clear();
	for(unsigned n = 0; n < net_layers.back().size() - 1; n++){
		results_values.push_back(net_layers.back()[n].get_output_value());
	}

}



net::net(const std::vector<unsigned> &net_topology){

	unsigned num_layers = net_topology.size();

	for(unsigned layer_index = 0; layer_index < num_layers; layer_index++){
		net_layers.push_back(layer());
		
		unsigned num_outputs = layer_index == net_topology.size() - 1? 0 : net_topology[layer_index + 1];

		// now we add the neurons to the layer
		// also include the bias neuron (which is why we use <=)
		for(unsigned neuron_index = 0; neuron_index <= net_topology[layer_index]; neuron_index++){
			
			// access the layer with .back()
			net_layers.back().push_back(neuron(num_outputs, neuron_index));
		}

		net_layers.back().back().set_output_value(1.0);
	}

}

// ******************** other functions ********************

void print_vector(std::string label, std::vector<double> &v){
	std::cout << label << " ";
	for(unsigned i = 0; i < v.size(); ++i){
		std::cout << v[i] << " ";
	}
	std::cout << std::endl;
}

void print_vector(std::string label, std::vector<unsigned> &v){
	std::cout << label << " ";
	for(unsigned i = 0; i < v.size(); ++i){
		std::cout << v[i] << " ";
	}
	std::cout << std::endl;
}

// ******************** class main ********************

int main(){

	std::cout<<"What is the training data?"<<std::endl;
	std::string data_path;
	std::cin>>data_path;
	training_data train_data(data_path);
	std::vector<unsigned> topology;
	train_data.get_topology(topology);
	print_vector("Topology: ", topology);
	net my_net(topology);

	std::vector<double> input_values, target_values, result_values;
	int training_pass = 0;
	while(!train_data.is_eof()){
		training_pass++;
		std::cout << std::endl << "Pass " << training_pass << std::endl;

		// Get new input data and feed it forward:
		if(train_data.get_next_inputs(input_values) != topology[0])
			break;
		print_vector("Inputs :", input_values);
		my_net.feed_forward(input_values);

		// Collect the net's actual results:
		my_net.get_results(result_values);
		print_vector("Outputs:", result_values);

		// Train the net what the outputs should have been:
		train_data.get_target_outputs(target_values);
		print_vector("Targets:", target_values);
		assert(target_values.size() == topology.back());

		my_net.back_propagate(target_values);

		// Report how well the training is working, average over recnet
		std::cout << "Net recent average error: "
		     << my_net.get_recent_average_error() << std::endl;
	}

	std::cout << std::endl << "Done" << std::endl;

	return 0;

}