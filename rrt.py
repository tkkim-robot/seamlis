from visibility_rrt.visibility_rrtStar import VisibilityRRTStar


x_start = (2.0, 2.0, 0)  # Starting node (x, y, yaw)
x_goal = (5.0, 3.0)  # Goal node

lqr_rrt_star = VisibilityRRTStar(x_start=x_start, x_goal=x_goal,
                                max_sampled_node_dist=1.0,
                                max_rewiring_node_dist=2,
                                goal_sample_rate=0.1,
                                rewiring_radius=2,  
                                iter_max=500,
                                solve_QP=False,
                                visibility=False,
                                collision_cbf=False,
                                show_animation=True)
waypoints, _ , _ = lqr_rrt_star.planning()