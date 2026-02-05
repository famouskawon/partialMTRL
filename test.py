# debug_mt10_env.py
import numpy as np
import metaworld_algorithms.envs

import gymnasium as gym

def main():
    # 1) registry 확인 (Meta-World namespace가 등록되어 있어야 함)
    keys = [k for k in gym.registry.keys() if "Meta-World" in k]
    print("=== Gym registry: Meta-World entries (first 50) ===")
    print(keys[:50])
    print("total:", len(keys))
    print()

    # 2) spec 확인 (entry_point가 곧 wrapper/구현의 시작점)
    spec = gym.spec("Meta-World/MT10")
    print("=== Spec for Meta-World/MT10 ===")
    print("id:", spec.id)
    print("entry_point:", spec.entry_point)
    print("kwargs:", spec.kwargs)
    print()

    # 3) vec env 생성
    envs = gym.make_vec(
        "Meta-World/MT10",
        seed=0,
        use_one_hot=True,
        terminate_on_success=False,
        max_episode_steps=200,
        vector_strategy="async",
        reward_function_version="v2",
        num_goals=50,
        reward_normalization_method=None,
        normalize_observations=False,
    )
    print("=== VecEnv created ===")
    print(envs)
    print("num_envs:", envs.num_envs)
    print("obs_space:", envs.observation_space)
    print("act_space:", envs.action_space)
    print()

    # 4) reset + 1 step
    obs, info0 = envs.reset()
    print("=== reset() ===")
    print("obs shape:", np.asarray(obs).shape)
    print("reset info keys:", list(info0.keys()) if isinstance(info0, dict) else type(info0))
    print()

    actions = envs.action_space.sample()
    next_obs, rewards, terminations, truncations, infos = envs.step(actions)
    done = np.logical_or(terminations, truncations)

    print("=== step() output shapes ===")
    print("next_obs:", np.asarray(next_obs).shape)
    print("rewards:", np.asarray(rewards).shape, "sample:", np.asarray(rewards)[:5])
    print("terminations:", np.asarray(terminations).shape, "sum:", int(np.sum(terminations)))
    print("truncations:", np.asarray(truncations).shape, "sum:", int(np.sum(truncations)))
    print("done sum:", int(np.sum(done)))
    print()

    # 5) infos 구조 덤프 (우리가 IL-RL 붙이려면 여기서 success/task id가 핵심)
    print("=== infos keys (top-level) ===")
    if isinstance(infos, dict):
        print(list(infos.keys()))
    else:
        print("infos is not dict:", type(infos))
        return
    print()

    # 6) 자주 쓰는 후보 키들 확인
    candidate_keys = [
        "success",
        "task_name",
        "task_id",
        "env_name",
        "task",
        "final_info",
        "final_obs",
    ]
    print("=== candidate keys check ===")
    for k in candidate_keys:
        if k in infos:
            v = infos[k]
            arr = np.asarray(v) if v is not None else None
            print(f"- {k}: type={type(v)} shape={getattr(arr,'shape',None)} sample={arr[:5] if arr is not None and arr.ndim>0 else arr}")
        else:
            print(f"- {k}: (missing)")
    print()

    # 7) final_info 상세 (episode 끝난 env가 있을 때만 의미있지만, 구조 확인용)
    if "final_info" in infos:
        fi = infos["final_info"]
        print("=== final_info structure ===")
        print("type:", type(fi))
        if isinstance(fi, dict):
            print("final_info keys:", list(fi.keys()))
            for kk in list(fi.keys())[:30]:
                vv = fi[kk]
                aa = np.asarray(vv) if vv is not None else None
                print(f"  - {kk}: type={type(vv)} shape={getattr(aa,'shape',None)}")
        else:
            try:
                print("len(final_info):", len(fi))
                print("first item:", fi[0])
            except Exception as e:
                print("cannot inspect final_info:", e)
        print()

    # 8) one-hot 확인 (obs 마지막 num_tasks가 one-hot인지)
    obs_arr = np.asarray(obs)
    # MT10이면 one-hot 길이 10
    oh = obs_arr[:, -10:]
    print("=== one-hot tail sanity check (last 10 dims) ===")
    print("one-hot shape:", oh.shape)
    print("row sums (first 10):", np.sum(oh[:10], axis=1))
    print("argmax (first 10):", np.argmax(oh[:10], axis=1))
    print("unique argmax:", np.unique(np.argmax(oh, axis=1)))
    print()

    envs.close()

if __name__ == "__main__":
    main()