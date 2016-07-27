# AIAnonymous Nash

## Problem
**Scenario** :
An user's session on a website, is an isolated set of actions.
There can be multiple such session of a user on the website.
In each session an user interacts with multiple objects (like product items, buttons, links, etc),
thereby creating a trail of activities.

**Aim**:
Using trails created by multiple users on the website.

1. Develop a model to understand users behavior on the website.
2. Predict user's value perception for any object on the website.

## Model
Dataset contains N number of variable trails. Each trail has a maximum length of M nodes.

Each node of the trail contains the following information

1. **Website Object (WO)**, user interacted with
2. **Action** user performed on the website object
3. **Reward** the website received from the user's action
4. **Expected User Payoff**, received thru that action.

#### Understanding user behavior
1. Identify groups of similar user behaviour, based on the interactions with WOs,
i.e. automatically cluster similar trails providing similar expected payoffs
to users and rewards to the website. This can be thought of as similar MDPs
of users.
2. Create probabilitics models of such cluster, which can be run as similations for
any simulated/chosen trail.
3. Model user's value perception of WOs, i.e, user's policy for choosing action, expected payoff
user perceive with the action.

2. Use model to predict user's action and expected payoff for a WO, in any simulated/chosen trail.