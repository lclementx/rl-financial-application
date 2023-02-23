from __future__ import annotations
from markov_process import *
import pprint as pprint

A = TypeVar('A')
S = TypeVar('S')
X = TypeVar('X')

# Reward processes
@dataclass(frozen=True)
class TransitionStep(Generic[S]):
    state: NonTerminal[S]
    next_state: State[S]
    reward: float

    def add_return(self, γ: float, return_: float) -> ReturnStep[S]:
        '''Given a γ and the return from 'next_state', this annotates the
        transition with a return for 'state'.
        '''
        return ReturnStep(
            self.state,
            self.next_state,
            self.reward,
            return_=self.reward + γ * return_
        )

@dataclass(frozen=True)
class ReturnStep(TransitionStep[S]):
    return_: float
    
#Now define a Markov Reward Process as an extension of the Markov Process --> 
class MarkovRewardProcess(MarkovProcess[S]):
    @abstractmethod
    def transition_reward(self, state: NonTerminal[S]) -> Distribution[Tuple[State[S], float]]:
        pass
    
    #Transition to get the next state's distribution --> sample to return next_state
    def transition(self, state: NonTerminal[S]) -> Distribution[State[S]]:
        distribution = self.transition_reward(state)
        def next_state(distribution=distribution):
            next_s, _ = distribution.sample()
            return next_s
        
        #return a SampledDistribution of the next_state
        return SampledDistribution(next_state)
    
    def simulate_reward(
        self,
        start_state_distribution: Distribution[NonTerminal[S]]
    ) -> Iterable[TransitionStep[S]]:
        state: State[S] = start_state_distribution.sample()
        reward: float = 0.
        
        while isinstance(state, NonTerminal):
            next_distribution = self.transition_reward(state)
            next_state, reward = next_distribution.sample()
            yield TransitionStep(state, next_state, reward)
            state = next_state
            
#State-Reward is the probability distribution of a given state and it's reward(s)
#This is used to define: for state_t --> (state_t+1 , reward_0): probability_0, 
#(state_t+1 , reward_1): probability_1, (state_t+1 , reward_2): probability_2 etc.
#Note: sum of all the probability_0,1 and 2 here will = probability state_t -> state_t+1
StateReward = FiniteDistribution[Tuple[State[S], float]]
RewardTransition = Mapping[NonTerminal[S], StateReward[S]]

class FiniteMarkovRewardProcess(FiniteMarkovProcess[S],
                                MarkovRewardProcess[S]):
    transition_reward_map: RewardTransition[S]
    reward_function_vec: np.ndarray
    
    def __init__(
        self,
        transition_reward_map: Mapping[S, FiniteDistribution[Tuple[S, float]]]
    ):
        transition_map: Dict[S, FiniteDistribution[S]] ={}
        for state, trans in transition_reward_map.items():
            probabilities: Dict[S, float] = defaultdict(float)
            for(next_state, _), probability in trans:
                #example: (next_state_0), reward_0): 0.3, (next_state_0, reward_1): 0.5, 
                #(next_state_1, reward_2): 0.2, 
                # = (next_state_0):0.8, (next_state_1): 0.2
                probabilities[next_state] += probability
                
            transition_map[state]=Categorical(probabilities)
            
        super().__init__(transition_map)
        
        #Do NonTerminal/Terminal wrapping for users
        nt: Set[S] = set(transition_reward_map.keys())
        self.transition_reward_map = {
            NonTerminal(s): Categorical(
                {(NonTerminal(s1) if s1 in nt else Terminal(s1), r): p
                 for (s1, r), p in v}
            ) for s, v in transition_reward_map.items()
        } 
        
        #Create vector - for a given state, find out what your expected reward is by looking at
        #the state you will transition into and get that reward * probability. Sum them up.
        self.reward_function_vec = np.array([
            sum(probability * reward for (_, reward), probability in
                self.transition_reward_map[state])
            for state in self.non_terminal_states
        ])
        
    #Fancy printing again
    def __repr__(self) -> str:
        display = ""
        for s, d in self.transition_reward_map.items():
            display += f"From State {s.state}:\n"
            for (s1, r), p in d:
                opt = "Terminal " if isinstance(s1, Terminal) else ""
                display +=\
                    f"  To [{opt}State {s1.state} and Reward {r:.3f}]"\
                    + f" with Probability {p:.3f}\n"
        return display
    
    #This return state reward (a finite Distribution). In a test example the transition_reward_map
    #returns a Categorical - finite Distribution. This is, a set of finite outcomes with its assigned
    #probabilities. e.g.
    #{
    #(state_1,10):0.3,
    #(state_2,8),0.1,
    #(state_2,5),0.2,
    #(state_3,6):0.5
    #} <-- this allows you to just sample the distribution to obtain the next state + reward
    def transition_reward(self, state: NonTerminal[S]) -> StateReward[S]:
        return self.transition_reward_map[state]
    
    #Value = Reward + Gamma * Transition Probability * Value
    #So Value = (Identity - gamma * Transition Probability)^-1 * Reward
    def get_value_function_vec(self, gamma: float) -> np.ndarray:
        inv = np.eye(len(self.non_terminal_states)) \
                - gamma * self.get_transition_matrix()
        return np.linalg.solve(
            inv,
            self.reward_function_vec
        )
    
    def display_reward_function(self):
        pprint.pprint({
            self.non_terminal_states[i]: round(r, 3)
            for i, r in enumerate(self.reward_function_vec)
        })

    def display_value_function(self, gamma: float):
        pprint.pprint({
            self.non_terminal_states[i]: round(v, 3)
            for i, v in enumerate(self.get_value_function_vec(gamma))
        })
