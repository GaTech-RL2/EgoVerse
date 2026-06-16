#ifndef CAN_H
#define CAN_H

#include "../libcan/SocketCAN.h"
#include "math_ops.h"
#include <array>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdint.h>
#include <string.h>

#define KP_MIN 0.0f
#define KP_MAX 500.0f
#define KD_MIN 0.0f
#define KD_MAX 5.0f
#define POS_MIN -12.5f
#define POS_MAX 12.5f
#define SPD_MIN -18.0f
#define SPD_MAX 18.0f
#define T_MIN -30.0f
#define T_MAX 30.0f
#define I_MIN -30.0f
#define I_MAX 30.0f

#define DM_P_MIN -12.5f // Radians
#define DM_P_MAX 12.5f
#define DM_V_MIN -45.0f // Rad/s
#define DM_V_MAX 45.0f
#define DM_KP_MIN 0.0f // N-m/rad
#define DM_KP_MAX 500.0f
#define DM_KD_MIN 0.0f // N-m/rad/s
#define DM_KD_MAX 5.0f
#define DM_T_MIN -18.0f
#define DM_T_MAX 18.0f

typedef struct
{
    int16_t current_actual_int;
    float speed_actual_rad;
    float angle_actual_rad;
    uint16_t motor_id;
    uint8_t temperature;
    uint8_t error;
    float angle_actual_float;
    float current_actual_float;

    float gripper_pos;
    float gripper_spd;
    float gripper_cur;
    float gripper_last_pos;
    float gripper_totalangle;
    float round_cnt;

} OD_Motor_Msg;

typedef struct CanPacket
{
    uint16_t StdId;
    uint8_t ExtId;
    uint8_t IDE;
    uint8_t RTR;
    uint8_t DLC;

    uint8_t Data[8];
} __attribute__((packed)) CanPacket;

typedef struct EcatTxPacket
{
    uint8_t LED;
    struct CanPacket can[2];
    uint8_t null;
} __attribute__((packed)) EcatTxPacket;

/// @brief ecat pdo recive data (slv to master)
typedef struct EcatRxPacket
{
    uint8_t switch_io;
    uint8_t null[5];

    struct CanPacket can[2];
} __attribute__((packed)) EcatRxPacket;

void RV_can_data_repack(uint32_t msgID, uint8_t *Data, int32_t databufferlen, uint8_t comm_mode,
                        std::array<OD_Motor_Msg, 10> &motor_msg);
void DM_can_data_repack(uint8_t *Data, std::array<OD_Motor_Msg, 10> &motor_msg);

class CanInterface
{
  public:
    virtual void transmit(can_frame_t &frame) = 0;
    const std::array<OD_Motor_Msg, 10> get_motor_msg();
    virtual bool is_open() = 0;
    void can_receive_frame(can_frame_t *frame);
};

class Usb2Can : public CanInterface
{
  public:
    Usb2Can(std::string interface_name);
    ~Usb2Can();

    void transmit(can_frame_t &frame) override;
    bool is_open() override;
};
class EtherCat2Can : public CanInterface
{
  public:
    EtherCat2Can(std::string interface_name);
    ~EtherCat2Can();

    void transmit(can_frame_t &frame) override;
    bool is_open() override;
};
class ArxCan
{
  public:
    ArxCan(std::string interface_name);
    ~ArxCan();

    void can_cmd_init(uint16_t motor_id, uint8_t cmd);

    void send_EC_motor_cmd(uint16_t motor_id, float kp, float kd, float pos, float spd, float tor);
    void query_EC_motor_pos(uint16_t motor_id);
    void query_EC_motor_vel(uint16_t motor_id);
    void query_EC_motor_current(uint16_t motor_id);
    void set_motor(uint16_t motor_id, uint8_t cmd);

    void send_DM_motor_cmd(uint16_t motor_id, float kp, float kd, float pos, float spd, float tor);
    void enable_DM_motor(uint16_t ID);
    void reset_zero_readout(uint16_t ID);
    void clear(uint16_t ID);

    const std::array<OD_Motor_Msg, 10> get_motor_msg();
};

#endif