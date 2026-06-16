#if (!defined(SOCKETCAN_H)) && (!defined(MINGW))
#define SOCKETCAN_H

// #include "CANAdapter.h"
#include "CANFrame.h"

#include <stdbool.h>
// IFNAMSIZ, ifreq
#include <net/if.h>
// Multi-threading
#include <pthread.h>

typedef enum {
  ADAPTER_NONE,
  ADAPTER_SOCKETCAN,
  ADAPTER_SLCAN,
  ADAPTER_LOGFILE,
} can_adapter_t;

/**
 * Interface request structure used for socket ioctl's
 */
typedef struct ifreq interface_request_t;

/**
 * Socket address type for CAN sockets
 */
typedef struct sockaddr_can can_socket_address_t;

typedef void (*reception_handler_t)(can_frame_t*, void*);

/**
 * Facilitates frame transmission and reception via a CAN adapter
 */
class SocketCAN {

 protected:
  can_adapter_t adapter_type;

 private:
  interface_request_t if_request;

  can_socket_address_t addr;

  pthread_t receiver_thread_id;

 public:
  void* reception_handler_data;
  reception_handler_t reception_handler;

  /**
     * CAN socket file descriptor
     */
  int sockfd = -1;

  /**
     * Request for the child thread to terminate
     */
  bool terminate_receiver_thread = false;

  bool receiver_thread_running = false;

  /** Constructor */
  SocketCAN();
  /** Destructor */
  ~SocketCAN();

  /**
     * Open and bind socket
     */
  void open(const char*);

  /**
     * Close and unbind socket
     */
  void close();

  /**
     * Returns whether the socket is open or closed
     *
     * @retval true     Socket is open
     * @retval false    Socket is closed
     */
  bool is_open();

  /**
     * Sends the referenced frame to the bus
     */
  void transmit(can_frame_t*);

  /**
     * Starts a new thread, that will wait for socket events
     */
  void start_receiver_thread();
};

#endif
