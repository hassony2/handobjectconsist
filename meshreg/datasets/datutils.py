def flip_hand_side(target_side, hand_side):
    # Flip if needed
    if target_side == "right" and hand_side == "left":
        flip = True
        hand_side = "right"
    elif target_side == "left" and hand_side == "right":
        flip = True
        hand_side = "left"
    else:
        flip = False
    return hand_side, flip
