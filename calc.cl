typedef struct {
    float x;
    float y;
} vector;

typedef struct {
    int iterations;
    float hole_radius;
    float hole1_x;
    float hole1_y;
    float hole2_x;
    float hole2_y;


} settings_data;

typedef struct {
    char a;
} out_data;

vector product(__private vector* v1, __private vector* v2);
vector sum(__private vector* v1, __private vector* v2);

bool magnitudegt(__private vector* v1, float mag);

vector product(__private vector* v1, __private vector* v2) {
    __private vector v = {v1->x * v2->x - v1->y*v2->y, v1->x * v2->y + v1->y * v2->x};
    return v;
}

vector sum(__private vector* v1, __private vector* v2) {
    __private vector v = {v1->x + v2->x, v1->y + v2->y};
    return v;
}

bool magnitudegt(__private vector* v1, float mag) {
    return v1->x * v1->x + v1->y * v1->y > mag * mag;
}


__kernel void check_inside(__global vector* positions,
                           __global settings_data* settings,
                           __global out_data* out

                          ) {



    __private int global_id = get_global_id(0);
    __private vector val = positions[global_id];
    __private vector c = positions[global_id];

    __private vector neg_hole_1 = {-settings->hole1_x, -settings->hole1_y};
    __private vector neg_hole_2 = {-settings->hole2_x, -settings->hole2_y};

    __private vector to_hole_1 = sum(&c, &neg_hole_1);
    __private vector to_hole_2 = sum(&c, &neg_hole_2);

    if (!magnitudegt(&to_hole_1, settings->hole_radius) || !magnitudegt(&to_hole_2, settings->hole_radius)) {
        out[global_id].a = 0;
        return;
    }

    for (int i = 0; i < settings->iterations; i++) {
        __private vector prod = product(&val, &val);
        val = sum(&prod, &c);

        if (magnitudegt(&val, 2)) {
            out[global_id].a = 0;
            return;
        }
    }
    out[global_id].a = 255;
    return;
}

