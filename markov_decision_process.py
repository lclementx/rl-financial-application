from __future__ import annotations
from markov_reward_process import *
from policy import *

A = TypeVar('A')
S = TypeVar('S')
X = TypeVar('X')

#State Space is finite, Action space is finite for each non-terminal state, the set of unique pairs
#of (NEXT_STATE,REWARD) transitions from each pair of current (non-terminal state + action) is finite --> FiniteMarkovDecisionProcess
StateReward = FiniteDistribution[Tuple[State[S],float]]
ActionMapping = Mapping[A, StateReward[S]]
StateActionMapping = Mapping[NonTerminal[S], ActionMapping[A, S]]

#Define Transition Step
@dataclass(frozen=True)
class TransitionStep(Generic[S, A]):
    state: NonTerminal[S]
    action: A
    next_state: State[S]
    reward: float
    
### If you evaluate a MDP with a specific Policy, you will obtain the implied MRP by the policy 
##For now, asssume: Discrete Time, Time-homogenous, countable spaces, countable transitions
class MarkovDecisionProcess(ABC, Generic[S, A]):
    #Given the non-terminal state, return all the actions possible (potentially infinite hence an iterable)
    @abstractmethod
    def actions(self, state:NonTerminal[S]) -> Iterable[A]:
        pass
    
    #Specify the distribution of pairs of next state nad reward, given a non-terminal
    #state and action.
    @abstractmethod
    def step(
        self,
        state: NonTerminal[S],
        action: A
    ) -> Distribution[Tuple[State[S], float]]:
        pass
    
    def apply_policy(self, policy: Policy[S, A]) -> MarkovRewardProcess[S]:
        mdp = self
        
        #Create a RewardProcess that implements the transition_reward function from MarkovRewardProcess
        #It will take in the policy that was passed in to this 'apply_policy'. It will sample the next state
        #and reward given the current state and SAMPLED action (from the policy).
        class RewardProcess(MarkovRewardProcess[S]):
            def transition_reward(
                self,
                state: NonTerminal[S]
            ) -> Distirbution[Tuple[State[S], float]]:
                actions: Distribution[A] = policy.act(state)
                return actions.apply(lambda a: mdp.step(state, a))
        return RewardProcess()
    
    
    def simulate_actions(
        self,
        start_states: Distribution[NonTerminal[S]],
        policy: Policy[S, A]
    ) -> Iterable[TransitionStep[S, A]]:
        state: State[S] = start_states.sample()
        
        while isisntance(state, NonTerminal):
            action_distribution = policy.act(state)
            action = action_distribution.sample()
            next_distribution = self.step(state,action)
            
            next_state, reward = next_distribution.sample()
            yield Transition(step, action, next_state, reward)
            state = next_state
    
class FiniteMarkovDecisionProcess(MarkovDecisionProcess[S, A]):
    mapping: StateActionMapping[S, A]
    non_terminal_states: Sequence[NonTerminal[S]]
    
    def __init__(
        self,
        mapping: Mapping[S, Mapping[A, FiniteDistribution[Tuple[S, float]]]]
    ):
        #Wrap Terminal and NonTerminal for the State Action mapping
        non_terminals: Set[S] = set(mapping.keys())
        self.mapping = {NonTerminal(s): {a: Categorical(
            {(NonTerminal(s1) if s1 in non_terminals else Terminal(s1), r): p
             for (s1, r), p in v}
        ) for a, v in d.items()} for s, d in mapping.items()}
        self.non_terminal_states = list(self.mapping.keys())
    
    #Fancy Printing
    def __repr__(self) -> str:
        display = ""
        for s, d in self.mapping.items():
            display += f"From State {s.state}:\n"
            for a, d1 in d.items():
                display += f" With Action {a}:\n"
                for (s1, r), p in d1:
                    opt = "Terminal " if isinstace(s1, Terminal) else ""
                    display += f"   To [{opt}State {s1.state} and "\
                        + f"Reward {r:3f}] with Probaility {p:.3f}\n"
            
            return display
    
    #Defining the step: given the state and action - Get the action map from State Action Mapping [NonTerminal[S] -> ActionMapping]
    #then us the action mapping to get the StateReward (finite Distribution of all the Next States + Rewards)
    def step(self, state: NonTerminal[S], action: A) -> StateReward[S]:
        action_map: ActionMapping[A, S] = self.mapping[state]
        return action_map[action]

    #Actions - given the State Action Mapping, return the action mapping and get all the keys i.e. Actions
    def actions(self, state:NonTerminal[S]) -> Iterable[A]:
        return self.mapping[state].keys()
    
    #Take a FinitePolicy as an input and return a FiniteMarkovRewardProcess by
    #Processing the finite structures of both MDP and the Policy and producing a finite structure
    #of the implied MRP
    def apply_finite_policy(self, policy:FinitePolicy[S, A])\
        -> FiniteMarkovRewardProcess[S]:
        
        transition_mapping: Dict[S, FioniteDistribution[Tuple[S, float]]] = {}
        for state in self.mapping:
            action_map: ActionMapping[A, S] = self.mapping[state]
            outcomes: DefaultDict[Tuple[S, float], float]\
                = defaultdict(float)
            actions = policy.act(state)
            for action, p_action in actions:
                for (s1, r), p in action_map[action]:
                    outcomes[(s1.state, r)] += p_action * p
            
            transition_mapping[state.state] = Categorical(outcomes)
        
        return FiniteMarkovRewardProcess(transition_mapping)