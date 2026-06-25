# todo

## question to myself

This is a question I had from yesterday: is the integration to bluesky objects good enough? Try one scenario, I will write down/copy the scenario myself. The Scenario should show the trajectory and the avoiding status based on the color.

What if someone wants to do the cd, cr, and reso in one algorithm, for instnace using reinforcement learning model. Can the existing structure accomodate it?

conf    = detect(obs, obs, rpz, hpz, dtlookahead)
conf_gt = detect(bs.traf, bs.traf, rpz, hpz, dtlookahead)

newtrack, newgs, newvs, alt = mvp.resolve(conf, obs, obs, cfg)
recovery_state, _ = resumenav_double_criteria(
    recovery_state, conf, obs, obs, active,
    id2idx=_id2idx, recover=_noop_recover,
)
action = (newtrack, newgs, newvs, alt, list(recovery_state.resopairs))

maybe those blocks above should be made into one function

cdarr(obs, cns, rpz, hpz, dtlookahead, detect, resolve, recovery) etc etc

then the cdarr can be replaced with something like:

rl(obs, cns, rpz, hpz, dtlookahead, detect, resolve, recovery)