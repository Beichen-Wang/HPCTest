#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include "Sock.h"
const char* books[] = {"War and Peace",
    "Pride and Prejudice",
    "The Sound and the Fury"};
void report(const char* msg, int terminate) {
    perror(msg);
    if (terminate) exit(-1); /* failure */
}

int main() {
    /* fd for the socket */
    int sockfd = socket(AF_INET, /* versus AF_LOCAL */
        SOCK_STREAM, /* reliable, bidirectional */
        0); /* system picks protocol (TCP) */
    if (sockfd < 0) report("socket", 1); /* terminate */
    /* get the address of the host */
    struct hostent* hptr = gethostbyname(Host); /* localhost: 127.0.0.1 */
    if (!hptr) report("gethostbyname", 1); /* is hptr NULL? */
    if (hptr->h_addrtype != AF_INET) /* versus AF_LOCAL */
        report("bad address family", 1);
    /* connect to the server: configure server's address 1st */
    struct sockaddr_in saddr;
    memset(&saddr, 0, sizeof(saddr));
    saddr.sin_family = AF_INET;
    saddr.sin_addr.s_addr = ((struct in_addr*) hptr->h_addr_list[0])->s_addr;
    saddr.sin_port = htons(PortNumber); /* port number in big-endian */
    if (connect(sockfd, (struct sockaddr*) &saddr, sizeof(saddr)) < 0)
    report("connect", 1);
    /* Write some stuff and read the echoes. */
    puts("Connect to server, about to write some stuff...");
    int i;
    for (i = 0; i < ConversationLen; i++) {
        if (write(sockfd, books[i], strlen(books[i])) > 0) {
            /* get confirmation echoed from server and print */
            char buffer[BuffSize + 1];
            memset(buffer, '\0', sizeof(buffer));
            if (read(sockfd, buffer, sizeof(buffer)) > 0)
                puts(buffer);
        }
    }
    puts("Client done, about to exit...");
    close(sockfd); /* close the connection */
    return 0;
}