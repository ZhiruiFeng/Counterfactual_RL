# Counterfactual_RL
Reinforcement Learning Agent that can make full use of counterfactual information.

Colonel rank (Creative Level)
<figure>
  <img src="http://fzruniverse.life/images/Colonel.png" alt="Colonel" height="40">
</figure>

**Backgrounds**

Imagine such a situation: there are three treasure chests in front of you, and you are only allowed to open one of them.  After opening one, you found there is nothing inside it. You may feel very upset. But, then I tell you one of the chests will burst after opening. Now, instead, you must feel so lucky. This is how counterfactual information changes our view of option value.

RL mostly like a loop of trial and learn, which will result in low learning speed and sampling inefficiency. And in real situations, it will also being easily to take destructive actions and being trapped in partial understanding of the environment. In fact, we have no idea how well the performance is when solely depends on rewards, because some envrionment will ofer a large mount of rewards. Especially, while using experience replay, a lot of past experiences will be out of date dramatically because of "Black Swan Effect".

**Functioning**

Thus this project aims at using countfactual information from human interactive input or seeing in multi agent environment acompany with personal experiences to get context value. And here we will use a quantitative framework to conceptualize the notion of trustworthiness, which will be used to combine information from different source. Then the context (or state) value sets the reference point to which an outcome should be compared before updating the option value, and update the rewards in experience replay memory.

## Implementation plan

- Step 1: Using some personal sampling collection as information from other agents, then extract context value from it.
- Step 2: Build a multi-agent environment to make it learn in real time. (It can be interesting if one agent is controled by human player as a mentor.)

Some RL model are proud of freeing from human interference, so it will not be used when comparing with baseline models. However, human interaction is allowed in real-world settings and can be very effective. Thus, it can be used in early stage testing.
