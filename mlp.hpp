#ifndef MLP_HPP
#define MLP_HPP
#include "algebra.hpp"

namespace alg {
	//@Class Multilayer Perceptron
	class MLP {
		private:
			class Sig {
				protected:
					D mn, mx;
				public:
					Sig(D const & p_min = 0, D const & p_max = 1.) :
					mn(p_min),
					mx(p_max) {}
					D operator()(D const & p_net) const {
						return mn + (mx - mn) / (1. + exp(-p_net));}};
			class DSig : public Sig {
				public:
					DSig(D const & p_min = 0, D const & p_max = 1.) :
					Sig(p_min, p_max) {}
					D operator()(D const & p_act) const {
						return .0001 + (mx - p_act) * (p_act - mn) / (mx - mn);}};
			VU         layer_sizes;
			VD         i;
			MD         o, d;
			TD         w;
			D          eta;
			Sig const  act;
			DSig const dact;
			// add a bias neuron of constant 1. to vector
			VD & addBias(VD & p_vec) const {
				p_vec.push_back(1.);
				return p_vec;}
			// remove the bias neuron from vector
			VD & removeBias(VD & p_vec) const {
				p_vec.pop_back();
				return p_vec;}
		public:
			MLP(std::initializer_list< UI > const & p_layer_sizes,
				D p_eta = .5,
				D const &  p_activation_min = 0., D const & p_p_activation_max = 1.,
				D const &  p_weights_min = 0.,    D const & p_weights_max = 1.,
				UI const & p_seed = time(nullptr)) :
			layer_sizes(p_layer_sizes.begin(), p_layer_sizes.end()),
			i(layer_sizes[0]),
			o(len(layer_sizes) - 1, VD()),
			d(len(layer_sizes) - 1, VD()),
			w(len(layer_sizes) - 1, MD()),
			eta(p_eta),
			act(p_activation_min, p_p_activation_max),
			dact(p_activation_min, p_p_activation_max) {
				shuffleWeights(p_weights_min, p_weights_max, p_seed);}
			void shuffleWeights(D const & p_min = -1., D const & p_max = 1., UI const & p_seed = time(nullptr)) {
				srand(p_seed);
				for(UI lyr = 0; lyr < len(w); ++ lyr)
					w[lyr] = p_min + (p_max - p_min) * mrnd< D >(layer_sizes[lyr + 1], layer_sizes[lyr] + 1);}
			void remember(VD const & p_pattern) {
				VD a = i = p_pattern;
				for(UI lyr = 0; lyr < len(w); ++lyr)
					o[lyr] = a = trnsfrm(w[lyr] | addBias(a), act);}
			void teach(VD const & p_teacher) {
				int lyr = len(o) - 1;
				VD  a = o[lyr] - p_teacher;
				while(0 <= lyr) {
					d[lyr] = trnsfrm(o[lyr], dact) * a;
					a = d[lyr] | w[lyr];
					removeBias(a);
					-- lyr;}
				a = i;
				D e = eta;// * exp(-1e-5 * loop);
				for(lyr = 0; static_cast<UI>(lyr) < len(w); ++ lyr) {
					w[lyr] -= e * d[lyr] ^ addBias(a);
					a = o[lyr];
					e *= 1.;}}
			D input(UI const & p_id) const {return i[p_id];}
			VD input() const {return i;}
			MD weights(UI const & p_lyr) const {return w[p_lyr];}
			TD weights() const {return w;}
			D output(UI const & p_id) const {return o[len(o) - 1][p_id];}
			VD output() const {return o[len(o) - 1];}
			D rms() const {
				return len(d) && len(d[len(d) - 1])
					? sqrt((d[len(d) - 1] | d[len(d) - 1]) / len(d[len(d) - 1]))
					: 0.;}};
	//@
}
#endif // MLP_HPP
