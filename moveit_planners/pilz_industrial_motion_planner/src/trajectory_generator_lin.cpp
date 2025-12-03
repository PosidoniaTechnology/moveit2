/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2018 Pilz GmbH & Co. KG
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Pilz GmbH & Co. KG nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

#include <pilz_industrial_motion_planner/trajectory_generator_lin.h>

#include <cassert>
#include <sstream>
#include <time.h>
#include <moveit/robot_state/conversions.h>
#include <kdl/path_line.hpp>
#include <kdl/trajectory_segment.hpp>
#include <kdl/utilities/error.h>
#include <tf2/convert.h>
#include <tf2_eigen_kdl/tf2_eigen_kdl.hpp>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace pilz_industrial_motion_planner
{
static const rclcpp::Logger LOGGER =
    rclcpp::get_logger("moveit.pilz_industrial_motion_planner.trajectory_generator_lin");
TrajectoryGeneratorLIN::TrajectoryGeneratorLIN(const moveit::core::RobotModelConstPtr& robot_model,
                                               const LimitsContainer& planner_limits, const std::string& /*group_name*/)
  : TrajectoryGenerator::TrajectoryGenerator(robot_model, planner_limits)
{
  if (!planner_limits_.hasFullCartesianLimits())
  {
    RCLCPP_ERROR(LOGGER, "Cartesian limits not set for LIN trajectory generator.");
    throw TrajectoryGeneratorInvalidLimitsException("Cartesian limits are not fully set for LIN trajectory generator.");
  }
}

void TrajectoryGeneratorLIN::extractMotionPlanInfo(const planning_scene::PlanningSceneConstPtr& scene,
                                                   const planning_interface::MotionPlanRequest& req,
                                                   TrajectoryGenerator::MotionPlanInfo& info) const
{
  RCLCPP_DEBUG(LOGGER, "Extract necessary information from motion plan request.");

  info.group_name = req.group_name;
  std::string frame_id{ robot_model_->getModelFrame() };

  // goal given in joint space
  if (!req.goal_constraints.front().joint_constraints.empty())
  {
    info.link_name = robot_model_->getJointModelGroup(req.group_name)->getSolverInstance()->getTipFrame();

    if (req.goal_constraints.front().joint_constraints.size() !=
        robot_model_->getJointModelGroup(req.group_name)->getActiveJointModelNames().size())
    {
      std::ostringstream os;
      os << "Number of joints in goal does not match number of joints of group "
            "(Number joints goal: "
         << req.goal_constraints.front().joint_constraints.size() << " | Number of joints of group: "
         << robot_model_->getJointModelGroup(req.group_name)->getActiveJointModelNames().size() << ")";
      throw JointNumberMismatch(os.str());
    }

    for (const auto& joint_item : req.goal_constraints.front().joint_constraints)
    {
      info.goal_joint_position[joint_item.joint_name] = joint_item.position;
    }

    // Ignored return value because at this point the function should always
    // return 'true'.
    computeLinkFK(scene, info.link_name, info.goal_joint_position, info.goal_pose);
  }
  // goal given in Cartesian space
  else
  {
    info.link_name = req.goal_constraints.front().position_constraints.front().link_name;
    if (req.goal_constraints.front().position_constraints.front().header.frame_id.empty() ||
        req.goal_constraints.front().orientation_constraints.front().header.frame_id.empty())
    {
      RCLCPP_WARN(LOGGER, "Frame id is not set in position/orientation constraints of "
                          "goal. Use model frame as default");
      frame_id = robot_model_->getModelFrame();
    }
    else
    {
      frame_id = req.goal_constraints.front().position_constraints.front().header.frame_id;
    }
    info.goal_pose = getConstraintPose(req.goal_constraints.front());
  }

  assert(req.start_state.joint_state.name.size() == req.start_state.joint_state.position.size());
  for (const auto& joint_name : robot_model_->getJointModelGroup(req.group_name)->getActiveJointModelNames())
  {
    auto it{ std::find(req.start_state.joint_state.name.cbegin(), req.start_state.joint_state.name.cend(), joint_name) };
    if (it == req.start_state.joint_state.name.cend())
    {
      std::ostringstream os;
      os << "Could not find joint \"" << joint_name << "\" of group \"" << req.group_name
         << "\" in start state of request";
      throw LinJointMissingInStartState(os.str());
    }
    size_t index = it - req.start_state.joint_state.name.cbegin();
    info.start_joint_position[joint_name] = req.start_state.joint_state.position[index];
  }

  // Ignored return value because at this point the function should always
  // return 'true'.
  computeLinkFK(scene, info.link_name, info.start_joint_position, info.start_pose);

  // check goal pose ik before Cartesian motion plan starts
  std::map<std::string, double> ik_solution;
  if (!computePoseIK(scene, info.group_name, info.link_name, info.goal_pose, frame_id, info.start_joint_position,
                     ik_solution))
  {
    std::ostringstream os;
    os << "Failed to compute inverse kinematics for link: " << info.link_name << " of goal pose";
    throw LinInverseForGoalIncalculable(os.str());
  }
}

void TrajectoryGeneratorLIN::plan(const planning_scene::PlanningSceneConstPtr& scene,
                                  const planning_interface::MotionPlanRequest& req,
                                  const MotionPlanInfo& plan_info,
                                  const double& sampling_time,
                                  trajectory_msgs::msg::JointTrajectory& joint_trajectory)
{
  const int count = req.num_planning_attempts;
  const double initial_dt = sampling_time;

  // Create Cartesian path once
  std::unique_ptr<KDL::Path> path(setPathLIN(plan_info.start_pose, plan_info.goal_pose));
  if (!path)
  {
    throw LinTrajectoryConversionFailure("Failed to create Cartesian LIN path",
                                         moveit_msgs::msg::MoveItErrorCodes::FAILURE);
  }

  for (int i = 0; i < count; ++i)
  {
    const bool is_last_pass = (i == count - 1);

    // ----------------------------
    // NEW SCALING FORMULAS
    // ----------------------------
    const double vel_scale = req.max_velocity_scaling_factor     / (1.0 + 0.5 * i);
    const double acc_scale = req.max_acceleration_scaling_factor / (1.0 + 0.5 * i);

    // Sampling time decreases slightly for smoother IK
    const double dt = initial_dt / (1.0 + 0.25 * i);

    // ----------------------------
    // Build velocity profile
    // ----------------------------
    std::unique_ptr<KDL::VelocityProfile> vp(
        cartesianTrapVelocityProfile(vel_scale, acc_scale, path));

    KDL::Trajectory_Segment cart_traj(path.get(), vp.get(), false);

    moveit_msgs::msg::MoveItErrorCodes error_code;

    bool success = generateJointTrajectory(scene,
                                           planner_limits_.getJointLimitContainer(),
                                           cart_traj,
                                           plan_info.group_name,
                                           plan_info.link_name,
                                           plan_info.start_joint_position,
                                           dt,
                                           joint_trajectory,
                                           error_code);

    if (success)
    {
      return;  // SUCCESS → exit early
    }

    // Last attempt failed → throw with diagnostics
    if (is_last_pass)
    {
      std::ostringstream os;
      os << "Failed to generate joint trajectory after " << count << " attempts.\n"
         << "Final velocity scaling: " << vel_scale << "\n"
         << "Final acceleration scaling: " << acc_scale << "\n"
         << "Final sampling time: " << dt;
      throw LinTrajectoryConversionFailure(os.str(), error_code.val);
    }

    // Otherwise retry with smaller v/acc and smaller dt
  }
}

std::unique_ptr<KDL::Path> TrajectoryGeneratorLIN::setPathLIN(const Eigen::Affine3d& start_pose,
                                                              const Eigen::Affine3d& goal_pose) const
{
  RCLCPP_DEBUG(LOGGER, "Set Cartesian path for LIN command.");

  KDL::Frame kdl_start_pose, kdl_goal_pose;
  tf2::transformEigenToKDL(start_pose, kdl_start_pose);
  tf2::transformEigenToKDL(goal_pose, kdl_goal_pose);
  double eqradius = planner_limits_.getCartesianLimits().getMaxTranslationalVelocity() /
                    planner_limits_.getCartesianLimits().getMaxRotationalVelocity();
  KDL::RotationalInterpolation* rot_interpo = new KDL::RotationalInterpolation_SingleAxis();

  return std::unique_ptr<KDL::Path>(
      std::make_unique<KDL::Path_Line>(kdl_start_pose, kdl_goal_pose, rot_interpo, eqradius, true));
}

}  // namespace pilz_industrial_motion_planner
