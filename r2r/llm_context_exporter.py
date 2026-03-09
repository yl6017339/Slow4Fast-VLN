import os
import sys
import json
import math
from typing import Dict, Any, List


def _default(obj):
    try:
        return float(obj)
    except Exception:
        return str(obj)


def export_episode_context(agent, traj: Dict[str, Any]):
    """
    Export a compact, LLM-friendly context of one episode.
    This function is offline-only and does not call any LLM.
    """
    args = agent.args
    out_dir = args.llm_export_dir or os.path.join(args.output_dir, 'llm_export')
    os.makedirs(out_dir, exist_ok=True)

    # basic identifiers
    instr_id = traj['instr_id']
    # current env provides observations/batch; we can re-query minimal info from agent.env/env.graphs
    # we collect graph neighbors for nodes appearing in trajectory
    # and a snapshot of final metrics if available in traj['details']

    # flatten trajectory to a sequence of vp ids
    vp_path: List[str] = []
    for seg in traj['path']:
        # each seg is list of viewpoints, append preserving order
        for vp in seg:
            if not vp_path or vp_path[-1] != vp:
                vp_path.append(vp)

    scan = None
    if hasattr(agent.env, 'batch') and agent.env.batch:
        # find corresponding item by instr_id
        for item in agent.env.batch:
            if item.get('instr_id') == instr_id:
                scan = item.get('scan')
                instruction = item.get('instruction')
                break
    else:
        instruction = None

    # neighbors and candidate map for nodes in path, and relative angles if available
    neighbors = {}
    rel_pose = {}
    
    
    if scan is not None and hasattr(agent.env, 'graphs') and scan in agent.env.graphs:
        G = agent.env.graphs[scan]
        for vp in vp_path:
            if vp in G:
                neighbors[vp] = list(G.neighbors(vp))
        
        # Calculate relative positions directly from the environment graph
        try:
            # Fix import path - it should be relative to the current module location
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from models.graph_utils import calculate_vp_rel_pos_fts
            
            # Extract node positions from the environment graph
            for vp in vp_path:
                rel_pose[vp] = {}
                if vp in G.nodes and 'position' in G.nodes[vp]:
                    vp_pos = tuple(G.nodes[vp]['position'])
                    for n in neighbors.get(vp, []):
                        if n in G.nodes and 'position' in G.nodes[n]:
                            n_pos = tuple(G.nodes[n]['position'])
                            rh, re, rd = calculate_vp_rel_pos_fts(
                                vp_pos, n_pos,
                                base_heading=0.0, base_elevation=0.0
                            )
                            rel_pose[vp][n] = {
                                'rel_heading': float(rh),
                                'rel_elev': float(re),
                                'dist': float(rd)
                            }
        except Exception as e:
            print(f"Warning: Failed to calculate relative positions from graph: {e}")
            # Fall back to default values

    # stepwise candidates (approximate): if agent.gmaps exists we can read last updated cand cache
    # keep minimal: for each distinct vp in path, include its neighbors as candidates with relative placeholders
    steps = []
    # Build Place indices (stable within this episode): first occurrence order
    place_idx = {}
    for vp in vp_path:
        if vp not in place_idx:
            place_idx[vp] = len(place_idx)
    
    # Also assign place IDs to all neighbors
    for vp in vp_path:
        for neighbor in neighbors.get(vp, []):
            if neighbor not in place_idx:
                place_idx[neighbor] = len(place_idx)

    def _wrap_pi(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _quantize_angle(a: float, step_deg: float = 30.0) -> float:
        step = math.radians(step_deg)
        return round(a / step) * step

    def direction_text(rel_heading: float, rel_elev: float,
                       elev_eps_deg: float = 5.0,   # 俯仰死区阈值
                       head_eps_deg: float = 5.0,   # 方位死区阈值
                       use_quantize: bool = True) -> str:
        # 1) 归一化到 [-pi, pi]
        h = _wrap_pi(rel_heading)
        e = rel_elev

        # 2) 可选量化到 30°
        if use_quantize:
            h = _quantize_angle(h, 30.0)
            e = _quantize_angle(e, 30.0)

        elev_eps = math.radians(elev_eps_deg)
        head_eps = math.radians(head_eps_deg)

        # 3) 俯仰优先但要过阈值
        if abs(e) > elev_eps:
            return 'go up' if e > 0 else 'go down'

        # 4) 平面判断：前进/左/右/掉头
        if abs(h) <= head_eps:
            return 'go forward'
        if abs(h) <= math.pi / 2:
            return 'turn right' if h > 0 else 'turn left'
        # 归一化后只会落在 <= pi，超过 90°就是掉头
        return 'turn around'

    for t, vp in enumerate(vp_path):
        cand_list = []
        
        # Add current viewpoint as a "stop" candidate
        stop_cand = {
            'vp': vp,
            'place': place_idx.get(vp, t),
            'rel_heading': 0.0,
            'rel_elev': 0.0,
            'dist': 0.0,
            'direction_text': 'stop',
            'is_current_vp': True
        }
        cand_list.append(stop_cand)
        
        # Add neighbor candidates
        for n in neighbors.get(vp, []):
            cand = {
                'vp': n, 
                'place': place_idx.get(n, len(place_idx)),  # fallback to next available ID
                'is_current_vp': False
            }
            if vp in rel_pose and n in rel_pose[vp]:
                # normalize/quantize values for stability in JSON
                h = _wrap_pi(rel_pose[vp][n]['rel_heading'])
                e = rel_pose[vp][n]['rel_elev']
                hq = _quantize_angle(h, 30.0)
                eq = _quantize_angle(e, 30.0)
                cand.update({
                    'rel_heading': float(hq),
                    'rel_elev': float(eq),
                    'dist': float(rel_pose[vp][n]['dist'])
                })
                # add direction text
                cand['direction_text'] = direction_text(h, e)
            else:
                # Add default values if rel_pose not available
                cand.update({
                    'rel_heading': 0.0,
                    'rel_elev': 0.0,
                    'dist': 1.0,  # default distance
                    'direction_text': 'go forward'
                })
            cand_list.append(cand)
            
        steps.append({
            't': t,
            'cur_vp': vp,
            'place': place_idx.get(vp, t),
            'candidates': cand_list
        })

    context = {
        'instr_id': instr_id,
        'scan': scan,
        'instruction': instruction,
        'trajectory_vpids': vp_path,
        'steps': steps,
        'success': None,
        'metrics': {},
    }

    # attach details if available
    if isinstance(traj.get('details'), dict):
        context['details'] = traj['details']

    # compute per-episode success & metrics if gt is available
    try:
        if scan is not None and hasattr(agent.env, 'gt_trajs') and instr_id in agent.env.gt_trajs:
            gt_scan, gt_path = agent.env.gt_trajs[instr_id]
            # Ensure scan matches (normally equal)
            use_scan = scan if scan is not None else gt_scan
            # env._eval_item expects pred_path as list of lists (segments)
            pred_path = traj['path']  # already list of lists in this codebase
            scores = agent.env._eval_item(use_scan, pred_path, gt_path)
            context['success'] = int(scores.get('success', 0))
            context['metrics'] = {
                'nav_error': float(scores.get('nav_error', 0.0)),
                'oracle_error': float(scores.get('oracle_error', 0.0)),
                'trajectory_lengths': float(scores.get('trajectory_lengths', 0.0)),
                'action_steps': int(scores.get('action_steps', 0)),
                'trajectory_steps': int(scores.get('trajectory_steps', 0)),
                'spl': float(scores.get('spl', 0.0)),
                'nDTW': float(scores.get('nDTW', 0.0)),
                'SDTW': float(scores.get('SDTW', 0.0)),
                'CLS': float(scores.get('CLS', 0.0)),
            }
            # fallback details for missing agent-provided details: add per-step goal distance & basic stats
            if 'details' not in context:
                details = {}
                shortest_distances = agent.env.shortest_distances[use_scan]
                goal = gt_path[-1]
                for st in steps:
                    vp = st['cur_vp']
                    try:
                        goal_dist = float(shortest_distances[vp][goal])
                    except Exception:
                        goal_dist = None
                    cand_dists = [c.get('dist', None) for c in st['candidates'] if not c.get('is_current_vp')]
                    best_cand = None  
                    best_cand_dist = None
                    for c in st['candidates']:
                        if c.get('is_current_vp'):
                            continue
                        d = c.get('dist', None)
                        if d is None:
                            continue
                        if best_cand_dist is None or d < best_cand_dist:
                            best_cand_dist = d
                            best_cand = c
                    details[vp] = {
                        'goal_dist': goal_dist,
                        'num_candidates': len(st['candidates']) - 1,
                        'best_cand_place': best_cand.get('place') if best_cand else None,
                        'best_cand_dist': float(best_cand_dist) if best_cand_dist is not None else None,
                    }
                context['details'] = details
    except Exception as e:
        # keep defaults if evaluation fails
        print(f"[warn] failed to attach episode metrics/details for {instr_id}: {e}")

    # write file
    out_path = os.path.join(out_dir, f'{instr_id}.json')
    with open(out_path, 'w') as f:
        json.dump(context, f, indent=2, default=_default)

    return out_path

