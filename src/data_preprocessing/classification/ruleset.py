import src.data_preprocessing.rule_engine as reng

FLIP_H = reng.Flip('horizontal')
FLIP_V = reng.Flip('vertical')
ROT_90 = reng.Rotate('90')
ROT_180 = reng.Rotate('180')
ROT_270 = reng.Rotate('270')
TRA_BACK = reng.Transpose('\\')
TRA_FRONT = reng.Transpose('/')


FULLY_SYMMETRIC_INFERENCES = [FLIP_H, FLIP_V, ROT_90, ROT_180, ROT_270, TRA_BACK, TRA_FRONT]
VERTICAL_SYMMETRIC_INFERENCES = [FLIP_V, ROT_180]
HORIZONTAL_SYMMETRIC_INFERENCES = [FLIP_H, ROT_180]
VERT_HOR_SYMMETRIC_INFERENCES = [FLIP_H, FLIP_V, ROT_180]


def spawn_rules_common_predicate(predicate, inferences):
    return [predicate.then(inference) for inference in inferences]


CLASSES = ['bend_left', 'bend_right', 'children', 'closed_both_directions',
           'give_way', 'hazard', 'left_only', 'no_entry', 'priority_road',
           'right_only', 'roadworks', 'roundabout', 'stop', 'straight_ahead_only',
           'straight_or_left_only', 'straight_or_right_only', 'traffic_lights']


RULESET = [reng.IfTrue().then(reng.Identity()),
           reng.IfClassIs('straight_or_left_only').then(FLIP_H + reng.ChangeLabel('straignt_or_right_only')),
           reng.IfClassIs('straight_or_right_only').then(FLIP_H + reng.ChangeLabel('straignt_or_left_only')),

           reng.IfClassIs('bend_left').then(FLIP_H + reng.ChangeLabel('bend_right')),
           reng.IfClassIs('bend_right').then(FLIP_H + reng.ChangeLabel('bend_left')),

           reng.IfClassIs('right_only').then(FLIP_H + reng.ChangeLabel('left_only')),
           reng.IfClassIs('right_only').then(ROT_180 + reng.ChangeLabel('left_only')),
           reng.IfClassIs('right_only').then(ROT_270 + reng.ChangeLabel('straight_ahead_only')),
           reng.IfClassIs('right_only').then(TRA_FRONT + reng.ChangeLabel('straight_ahead_only')),
           reng.IfClassIs('right_only').then(FLIP_V),

           reng.IfClassIs('left_only').then(FLIP_H + reng.ChangeLabel('right_only')),
           reng.IfClassIs('left_only').then(ROT_180 + reng.ChangeLabel('right_only')),
           reng.IfClassIs('left_only').then(ROT_90 + reng.ChangeLabel('straight_ahead_only')),
           reng.IfClassIs('left_only').then(TRA_BACK + reng.ChangeLabel('straight_ahead_only')),
           reng.IfClassIs('left_only').then(FLIP_V),

           reng.IfClassIs('straight_ahead_only').then(ROT_90 + reng.ChangeLabel('right_only')),
           reng.IfClassIs('straight_ahead_only').then(ROT_270 + reng.ChangeLabel('left_only')),
           reng.IfClassIs('straight_ahead_only').then(TRA_FRONT + reng.ChangeLabel('right_only')),
           reng.IfClassIs('straight_ahead_only').then(TRA_BACK + reng.ChangeLabel('left_only')),
           reng.IfClassIs('straight_ahead_only').then(FLIP_H),

           reng.IfClassIs('traffic_lights').then(FLIP_H),
           reng.IfClassIs('give_way').then(FLIP_H),
           reng.IfClassIs('hazard').then(FLIP_H)
          ]
RULESET = (
        RULESET
        + spawn_rules_common_predicate(reng.IfClassIs('priority_road'), FULLY_SYMMETRIC_INFERENCES)
        + spawn_rules_common_predicate(reng.IfClassIs('closed_both_directions'), FULLY_SYMMETRIC_INFERENCES)
        + spawn_rules_common_predicate(reng.IfClassIs('no_entry'), VERT_HOR_SYMMETRIC_INFERENCES)
        + spawn_rules_common_predicate(reng.IfClassIs('roundabout'), FULLY_SYMMETRIC_INFERENCES)
)