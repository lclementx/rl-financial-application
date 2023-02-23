from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, replace, field
import itertools
import numpy as np
from typing import (Callable, Dict, Generic, Iterator, Iterable, List,
                    Mapping, Optional, Sequence, Tuple, TypeVar, overload)
import iterate as iterate
import numpy as np

X = TypeVar('X')
F = TypeVar('F', bound='FunctionApprox')

class FunctionApprox(ABC, Generic[X]):
    @abstractmethod
    def __add__(self: F, other: F) -> F:
        pass
    
    @abstractmethod
    def __mul__(self: F, other: F) -> F:
        pass
    
    #Compute the gradient of an objective function (cross_entropy loss?) of the function approximation with resepct to parameters in the internal representation of the function approximation
    @abstractmethod
    def objective_gradient(
        self: F,
        xy_vals_seq: Iterator[Tuple[X, float]],
        obj_deriv_out_fun: Callable[[Sequence[X], Sequence[float]], np.ndarray]
    )-> Gradient[F]:
        pass
    
    @abstractmethod
    def evaluate(self, x_values_seq: Iterable[X]) -> np.ndarray:
        pass
    
    #Take a Gradient and update the internal parameters using the gradient values e.g. gradient descent update to the params
    @abstractmethod
    def update_with_gradient(
        self:F,
        gradient: Gradient[F]
    ) -> F:
        pass
    
    def update(
        self: F,
        xy_vals_seq: Iterable[Tuple[X, float]]
    ) -> F:
        '''Update the internal parameters of the FunctionApprox
        based on incremental data provided in the form of (x,y)
        paris as a xy_vals_seq data structure
        '''
        
        #Think this is only applicable for cross entropy loss
        def deriv_func(x: Sequence[X], y: Sequence[float]) -> np.ndarray:
            return self.evaluate(x) - np.array(y)
        
        return self.update_with_gradient(
            self.objective_gradient(xy_vals_seq, deriv_func)
        )

    @abstractmethod
    def solve(
        self: F,
        xy_vals_seq: Iterable[Tuple[X, float]],
        error_tolerance: Optional[float] = None
    )-> F:
        pass
    
    #Ask the question: are 2 function approx close enough?
    @abstractmethod
    def within(self: F, other: F, tolerance: float) -> bool:
        pass
    
    def iterate_updates(
        self: F,
        xy_seq_stream: Iterator[Iterable[Tuple[X, float]]]
    ) -> Iterator[F]:
        return iterate.accumulate(
            xy_seq_stream,
            lambda fa, xy: fa.update(xy),
            initial=self
        )
    
    #Root mean square error - basis to check how good the FunctionApproximation is
    def rmse(
        self, 
        xy_vals_seq: Iterable[Tuple[X, float]]
    )-> float:
        x_seq, y_seq = zip(*xy_vals_seq)
        errors: np.ndarray = self.evaluate(x_seq) - np.array(y_seq)
        return np.sqrt(np.mean(errors * errors))
    
    #Return the x value that maximizes E_f(x;w)[y]
    def argmax(self,xs: Iterable[X]) -> X:
        args: Sequence[X] = list(xs)
        return args[np.argmax(self.evaluate(args))]
    
#Has F in this so that you can represent gradient with the internal parameters of the function approximation
@dataclass(frozen=True)
class Gradient(Generic[F]):
    function_approx: F
    
    @overload
    def __add__(self, x: Gradient[F]) -> Gradient[F]:
        ...
    
    @overload
    def __add__(self, x: F) -> F:
        ...
    
    def __add__(self, x):
        if isinstance(x, Gradient):
            return Gradient(self.function_approx + x.function_approx)
        
        return self.function_approx + x
    
    def __mul__(self: Graient[F], x: float) -> Gradient[F]:
        return Gradient(self.function_approx * x)
    
    def zero(self) -> Gradient[F]:
        return Gradient(self.function_approx * 0.)

SMALL_NUM = 1e-6

@dataclass(frozen=True)
class AdamGradient:
    learning_rate:float
    decay1: float
    decay2: float
    
    @staticmethod
    def default_setting() -> AdamGradient:
        return AdamGradient(
            learning_rate=0.001,
            decay1=0.9,
            decay2=0.999
        )

@dataclass(frozen=True)
class Weights:
    adam_gradient: AdamGradient
    time: int
    weights: np.ndarray
    adam_cache1: np.ndarray
    adam_cache2: np.ndarray
    
    @staticmethod
    def create(
        weights: np.ndarray,
        adam_cache1: Optional[np.ndarray] = None,
        adam_cache2: Optional[np.ndarray] = None,
        adam_gradient: AdamGradient = AdamGradient.default_setting()
    ) -> Weights:
        return Weights(
            adam_gradient=adam_gradient,
            time=0,
            weights=weights,
            adam_cache1=np.zeros_like(weights) if adam_cache1 is None else adam_cache1,
            adam_cache2=np.zeros_like(weights) if adam_cache2 is None else adam_cache2
        )
    
    #Idea of ADAM (Adaptive Moments) is that you use the a geometrically decaying weight for the previous learning rates and gradients to speed up convergence
    def update(self, gradient: np.array) -> Weights:
        time: int = self.time + 1
        #Cache for past gradients to adjudst adpative moments
        new_adam_cache1: np.ndarray = self.adam_gradient.decay1 * \
            self.adam_cache1 + (1 - self.adam_gradient.decay1) * gradient
        new_adam_cache2: np.ndarray = self.adam_gradient.decay2 * \
            self.adam_cache2 + (1 - self.adam_gradient.decay2) * gradient ** 2
        corrected_m: np.ndarray = new_adam_cache1 / \
            (1 - self.adam_gradient.decay1 ** time)
        corrected_v: np.ndarray = new_adam_cache2 / \
            (1 - self.adam_gradient.decay2 ** time)
        
        new_weights: np.ndarray = self.weights - \
            self.adam_gradient.learning_rate * corrected_m / \
            (np.sqrt(corrected_v) + SMALL_NUM) #SMALL NUM FOR LAPLACE SMOOTHING
        
        return replace(
            self,
            time=time,
            weights=new_weights,
            adam_cache1=new_adam_cache1,
            adam_cache2=new_adam_cache2,
        )
    
    def within(self, other: Weights[X], tolerance: float) -> bool:
        return np.all(np.abs(self.weights - other.weights) <= tolerance).item()
    
@dataclass(frozen=True)
class LinearFunctionApprox(FunctionApprox[X]):
    feature_functions: Sequence[Callable[[X], float]]
    regularization_coeff: float
    weights: Weights
    direct_solve: bool
    
    def __add__(self, other: LinearFunctionApprox[X]) -> \
        LinearFunctionApprox[X]:
        return replace(
            self,
            weights=replace(
                self.weights,
                weights=self.weights.weights + other.weights.weights
            )
        )

    def __mul__(self, scalar: float) -> LinearFunctionApprox[X]:
        return replace(
            self,
            weights=replace(
                self.weights,
                weights=self.weights.weights * scalar
            )
        )

    def evaluate(self, x_values_seq: Iterable[X]) -> np.ndarray:
        return np.dot(
            self.get_feature_values(x_values_seq),
            self.weights.weights
        )
    
    @staticmethod
    def create(
        feature_functions: Sequence[Callable[[X], float]],
        regularization_coeff: float = 0.,
        weights: Optional[Weights] = None,
        direct_solve: bool = True,
        adam_gradient: AdamGradient = AdamGradient.default_setting(),
    ) -> LinearFunctionApprox[X]:
        return LinearFunctionApprox(
            feature_functions=feature_functions,
            regularization_coeff=regularization_coeff,
            weights=Weights.create(
                adam_gradient=adam_gradient,
                weights=np.zeros(len(feature_functions))
            ) if weights is None else weights,
            direct_solve=direct_solve
        )
    
    def get_feature_values(self, x_values_seq: Iterable[X]) -> np.ndarray:
        return np.array(
            [[f(x) for f in self.feature_functions] for x in x_values_seq]
        )
    
    #Updating the weights: the mean of the feature vectors, weighted by the scalar linear prediction erros (+ regularization).
    def objective_gradient(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        obj_deriv_out_fun: Callable[[Sequence[X], Sequence[float]],float]
    ) -> Gradient[LinearFunctionApprox[X]]:
        x_vals, y_vals = zip(*xy_vals_seq)
        obj_deriv_out: np.array = obj_deriv_out_fun(x_vals,y_vals)
        features: np.ndarray = self.get_feature_values(x_vals)
        gradient: np.ndarray = \
            features.T.dot(obj_deriv_out) / len(obj_deriv_out) \
            + self.regularization_coeff * self.weights.weights
        return Gradient(replace(
            self,
            weights=replace(self.weights,weights=gradient)
        ))
            
    def update_with_gradient(
        self,
        gradient: Gradient[LinearFunctionApprox[X]]
    ) -> LinearFunctionApprox[X]:
        return replace(
            self,
            weights=self.weights.update(
                gradient.function_approx.weights.weights
            )
        )
    
    def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
        if isinstance(other, LinearFunctionApprox):
            return self.weights.within(other.weights, tolerance)
        else:
            return False
    
    def solve(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        error_tolerance: Optional[float] = None
    ) -> LinearFunctionApprox[X]:
        if self.direct_solve:
            x_vals, y_vals = zip(*xy_vals_seq)
            feature_vals: np.ndarray = self.get_feature_values(x_vals)
            feature_vals_t: np.ndarray = feature_vals.T
            left: np.ndarray = np.dot(feature_vals.T, feature_vals)\
                + feature_vals.shape[0] * self.regularization_coeff * \
                np.eye(len(self.weights.weights))
            right: np.ndarray = np.dot(feature_vals_t, y_vals)
            ret = replace(
                self,
                weights=Weights.create(
                    adam_gradient=self.weights.adam_gradient,
                    weights=np.linalg.solve(left,right)
                )
            )
        else:
            tol: float = 1e-6 if error_tolerance is None else error_tolerance
            def done(
                a: LinearFunctionApprox[X],
                b: LinearFunctionApprox[X],
                tol: float = tol
            ) -> bool:
                return a.within(b, tol)
            
            ret = iterate.converged(
                self.iterate_updates(itertools.repeat(list(xy_vals,seq))),
                done=done
            )
        return ret

##Deep NeuroNetwork Approximation
@dataclass(frozen=True)
class DNNSpec:
    neurons: Sequence[int] #number of neurons in the layer
    bias: bool #bias?
    hidden_activation: Callable[[np.ndarray], np.ndarray]
    hidden_activation_deriv: Callable[[np.ndarray], np.ndarray]
    output_activation: Callable[[np.ndarray], np.ndarray]
    output_activation_deriv: Callable[[np.ndarray], np.ndarray]
    
@dataclass(frozen=True)
class DNNApprox(FunctionApprox[X]):
    feature_functions: Sequence[[Callable[X], float]]
    dnn_spec: DNNSpec
    regularization_coeff: float
    weights: Sequence[Weights]
    
    def __add__(self, other: DNNApprox[X]) -> DNNApprox[X]:
        return replace(
            self,
            weights=[replace(w, weights=w.weights + o.weights) for
                     w, o in zip(self.weights, other.weights)]
        )

    def __mul__(self, scalar: float) -> DNNApprox[X]:
        return replace(
            self,
            weights=[replace(w, weights=w.weights * scalar)
                     for w in self.weights]
        )
    @staticmethod
    def create(
        feature_functions: Sequence[Callable[[X], float]],
        dnn_spec: DNNSpec,
        regularization_coeff: float=0.,
        weights: Optional[Sequence[Weights]] = None,
        adam_gradient: AdamGradient = AdamGradient.default_setting(),
    ) -> DNNApprox[X]:
        #Initialize Weights if its not specified.
        if weights is None:
            inputs: Sequence[int] = [len(feature_functions)] + \
                [n + (1 if dnn_spec.bias else 0)
                for i, n in enumerate(dnn_spec.neurons)] #based on the number of neurons in each layer, i = layer, n = number of neurons
            outputs: Seqeunce[int] = list(dnn_spec.neurons) + [1]
            wts = [Weights.create(
                weights=np.random.randn(output, inp) / np.sqrt(inp),
                adam_gradient=adam_gradient
            ) for inp, output in zip(inputs, outputs)]
        else:
            wts = weights
        return DNNApprox(
            feature_functions=feature_functions,
            dnn_spec=dnn_spec,
            regularization_coeff=regularization_coeff,
            weights=wts
        )
    
    def update_with_gradient(
        self,
        gradient: Gradient[DNNApprox[X]]
    ) -> DNNApprox[X]:
        return replace(
            self,
            weights=[w.update(g.weights) for w, g in
                     zip(self.weights, gradient.function_approx.weights)]
        )
    
    def get_feature_values(self, x_values_seq: ITerable[X]) -> np.ndarray:
        return np.array(
            [[f(x) for f in self.feature_functions] for x in x_values_seq]
        )

    def forward_propagation(
        self,
        x_values_seq: Iterable[X]
    ) -> Sequence[np.ndarray]:
        #x_values_seq: a n-length iterbale of input points
        #return: list of length L+2 where the first L+1 values
        #each represent the 2-D input arrays (of size n x |i_L|),
        #for each of the (L+1) layers (L of which are hidden layers),
        #and the last value represents the output of the DNN (As a 
        #1-D array of length n
        
        inp: np.ndarray = self.get_feature_values(x_values_seq)
        ret: List[np.ndarray] = [inp] 
        for w in self.weights[:-1]:
            out: np.ndarray = self.dnn_spec.hidden_activation(
                np.dot(inp,w.weights.T)
            )
            if self.dnn_spec.bias:
                inp = np.insert(out,0,1.,axis=1)
            else:
                inp = out
        
            ret.append(inp)
        ret.append(
            self.dnn_spec.output_activation(
                np.dot(inp, self.weights[-1].weights.T)
            )[:, 0]
        )
        return ret
    
    def evaluate(self, x_values_seq: Iterable[X]) -> np.ndarray:
        return self.forward_propagation(x_values_seq)[-1]
    
    def backward_propagation(
        self,
        fwd_prop: Sequence[np.ndarray],
        obj_deriv_out: np.ndarray
    )-> Sequence[np.ndarray]:
        '''
        fwd_prop represents the result of forward prop without final output, a sequence of L 2-D np.ndarrays of the DNN
        obj_der-v_out represents the derivative of the obejctive function with respect to the linear predictor of the final layer
        
        return: list (of length L+1) of |o_l| x |i_l| 2-D arrays, i.e. same as the type of self.weights.weights
        This function computes the gradient (with respect to weights) of thje objective where the output layer activation
        function is the canonical link function of the conditional distirbution y|x
        
        this all kinda assumes cross entropy loss :)
        '''
        
        deriv: np.ndarray = obj_deriv_out.reshape(1,-1) #Make it a columnm
        back_prop: List[np.ndarray] = [np.dot(deriv, fwd_prop[-1])/deriv.shape[1]]
        
        '''
        L is the numnber of hidden layers, n is the bunber of points
        layer l deriv represents dObjs/ds_l where s_l = i_l dot weights_l
        (s_l is the result of applying layer l without the activation function
        '''
        for i in reversed(range(len(self.weights)-1)):
            '''
            deriv_l is a 2-D array of dimension |o_l| x_n
            The recurvisve formulation of deriv is as follows:
            deriv_{l-1} = (weights_l.T inner deriv_l) haddamard g'(s_{l-1}),
            which is ((|i_l| x |o_l|) inner (|o_l| xn)) haddamard
            (|i_l| x n), which is (|i_l| x n ) = (|o_{l-1}| x n)
            Note: g'(s_{l_1}) is expressed as hidden layer activation
            derivative as a function of o_{l-1} 
            '''
            deriv = np.dot(self.weights[i+1].weights.T, deriv) * \
                self.dnn_spec.hidden_activation_deriv(fwd_prop[i+1].T)
            '''
            If self.dnn_spec.bias is True, then i_l = o_{l-1} + 1, in which
            case # the first row of the calculated deriv is removed to yield
            a 2-D array of dimension |o_{l-1}| x n.
            '''
            if self.dnn_spec.bias:
                deriv = deriv[1:]
            
            '''
            layer l gradient is deriv_l inner fwd_prop[l], which is
            of dimension (|o_l| x n) inner (n x |i_l|) = |o_l| x |i_l|)
            '''
            back_prop.append(np.dot(deriv, fwd_prop[i]) / deriv.shape[1])
        return back_prop[::-1]
    
    def objective_gradient(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        obj_deriv_out_fun: Callable[[Sequence[X], Sequence[float]], float]
    ) -> Gradient[DNNApprox[X]]:
        x_vals, y_vals = zip(*xy_vals_seq)
        obj_deriv_out: np.ndarray = obj_deriv_out_fun(x_vals, y_vals)
        fwd_prop: Sequence[np.ndarray] = self.forward_propagation(x_vals)[:-1]
        gradient: Sequence[np.ndarray] = \
            [x + self.regularization_coeff * self.weights[i].weights
             for i, x in enumerate(self.backward_propagation(
                 fwd_prop=fwd_prop,
                 obj_deriv_out=obj_deriv_out
             ))]
        return Gradient(replace(
            self,
            weights=[replace(w, weights=g) for \
                    w, g in zip(self.weights, gradient)]
        ))
    
    def solve(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        error_tolerance: Optional[float] = None
    ) -> DNNApprox[X]:
        tol: float = 1e-6 if error_tolerance is None else error_tolerance
        
        def done(
            a: DNNApprox[X],
            b: DNNApprox[X],
            tol: float = tol
        ) -> bool:
            return a.within(b, tol)
        
        return iterate.converged(
            self.iterate_updates(itertools.repeat(list(xy_vals_seq))),
            done=done
        )
    
    def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
        if isinstance(other, DNNApprox):
            return all(w1.within(w2, tolerance)
                       for w1, w2 in zip(self.weights, other.weights))
        else:
            return False
        
# @dataclass(frozen=True)
# class Tabular(FunctionApprox[X]):
#     values_map: Mapping[X, float] = field(default_factory=lambda: {})
#     counts_map: Mapping[X, int] = field(default_factory=lambda: {})
#     count_to_weight_func: Callable[[int], float] = \
#         field(default_factor
                                