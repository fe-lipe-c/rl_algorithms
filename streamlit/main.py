import pandas as pd
import numpy as np
import streamlit as st
from base.render_setup import render_sidebar, page_style

page_style()


def run_interface():
    page_title = "REINFORCEMENT LEARNING"
    st.markdown(
        f"""
        <div style="color: {st.session_state.text_color}; font-size: 1.1rem; text-align: justify; margin-top: 20px; padding: 50px;">
            <h4 style="color: #c7cc45;">{page_title}</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("---")

    st.markdown(
        r"""
# Reinforcement Learning Algorithms

## Policy Gradient Methods

In policy-based RL, the policy is a function $\pi_{\theta}$, where $\theta$ are the parameters. 

###  A2C (Advantage Actor-Critic)

A2C (Advantage Actor-Critic) is a policy gradient algorithm that combines the actor-critic architecture with advantage functions to improve sample efficiency and stability in reinforcement learning. A2C belongs to the family of actor-critic methods, which maintain two components:

- An **actor** (policy network) that decides which actions to take
- A **critic** (value network) that evaluates how good those actions are

The key innovation in A2C is using the "advantage function" to reduce variance in policy gradient updates while maintaining an unbiased estimate of the gradient.

The foundation of A2C is the policy gradient theorem. For a policy $\pi_{\theta}$ parameterized by $\theta$, the gradient of the expected return $J(\theta)$ is:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R_t \right]$$

Where:
- $\tau$ is a trajectory $(s_0, a_0, r_0, s_1, a_1, r_1, \dots)$
- $R_{t}$ is the return (discounted sum of rewards) from time t
                """
    )


def init_session():
    pass


def main():
    init_session()
    render_sidebar()
    run_interface()


if __name__ == "__main__":
    main()
