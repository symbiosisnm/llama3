from core.modules import HModule, LModule
from core.controller import ReasoningController
from core.memory import EpisodicMemory
from core.thought import Thought, ThoughtBuffer
from utils.validation import validate_input
from utils.config import HRMConfig
import torch


def orchestrate_reasoning(task: str, context: torch.Tensor, config: HRMConfig, device: str = 'cpu'):
    """
    Run an HRM reasoning episode given a task and input context.

    Args:
        task: Description of the task being solved.
        context: Input tensor of shape [batch, context_dim].
        config: HRMConfig specifying module dimensions and parameters.
        device: Torch device to run modules on.

    Returns:
        A dictionary with final high-level and low-level latents, the trace of latent states,
        and diagnostics including convergence information.
    """
    # Validate the context shape (allow flexible batch dimension)
    expected_shape = (context.shape[0], config.context_dim)
    validate_input(context, expected_shape)

    # Instantiate modules according to config
    hmod = HModule(config.context_dim, config.h_hidden_dim, config.h_plan_dim).to(device)
    lmod = LModule(config.h_plan_dim, config.l_hidden_dim).to(device)

    # Create the reasoning controller
    controller = ReasoningController(hmod, lmod, cycles=config.cycles, max_l_steps=config.max_l_steps, tol=config.tol)

    # Run the hierarchical reasoning process
    h_final, l_final, trace, diagnostics = controller.run(context.to(device))

    # Build a thought buffer to record each step
    buffer = ThoughtBuffer()
    buffer.add(Thought(content=f"Task: {task}", step_type='task'))
    buffer.add(Thought(content=f"Final H latent: {h_final}", step_type='state'))
    buffer.add(Thought(content=f"Final L latent: {l_final}", step_type='state'))
    for entry in trace:
        cycle_num = entry['cycle']
        buffer.add(Thought(content=f"Cycle {cycle_num} latents recorded", step_type='trace'))

    # Log the episode in episodic memory
    episode = EpisodicMemory()
    episode.log({
        'task': task,
        'context': context.detach().cpu().tolist(),
        'trace': buffer.to_list(),
        'diagnostics': diagnostics
    })

    return {
        'final_state': (h_final, l_final),
        'trace': buffer.to_list(),
        'diagnostics': diagnostics
    }
