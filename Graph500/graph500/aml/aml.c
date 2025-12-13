/* Copyright (c) 2011-2017 Graph500 Steering Committee
   All rights reserved.
   Developed by:                Anton Korzh anton@korzh.us
                                Graph500 Steering Committee
                                http://www.graph500.org
   New code under University of Illinois/NCSA Open Source License
   see license.txt or https://opensource.org/licenses/NCSA
*/

// AML: active messages library v 1.0
// MPI-3 passive transport
// transparent message aggregation greatly increases message-rate for loosy interconnects
// shared memory optimization used
// Implementation basic v1.0

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <ccutils/mpi/mpi_macros.hpp>

#ifdef VERBOSE_PRINTS
#include <assert.h>
#endif

#ifdef __APPLE__
#define SYSCTL_CORE_COUNT   "machdep.cpu.core_count"
#include <sys/sysctl.h>
#include <sys/types.h>
#include <mach/thread_policy.h>
#include <mach/thread_act.h>
// code borrowed from http://yyshen.github.io/2015/01/18/binding_threads_to_cores_osx.html
typedef struct cpu_set {
  uint32_t    count;
} cpu_set_t;

static inline void
CPU_ZERO(cpu_set_t *cs) { cs->count = 0; }

static inline void
CPU_SET(int num, cpu_set_t *cs) { cs->count |= (1 << num); }

static inline int
CPU_ISSET(int num, cpu_set_t *cs) { return (cs->count & (1 << num)); }

int sched_getaffinity(pid_t pid, size_t cpu_size, cpu_set_t *cpu_set)
{
  int32_t core_count = 0;
  size_t  len = sizeof(core_count);
  int ret = sysctlbyname("machdep.cpu.core_count", &core_count, &len, 0, 0);
  if (ret) {
    printf("error while get core count %d\n", ret);
    return -1;
  }
  cpu_set->count = 0;
  for (int i = 0; i < core_count; i++) {
    cpu_set->count |= (1 << i);
  }

  return 0;
}
int pthread_setaffinity_np(pthread_t thread, size_t cpu_size,
                           cpu_set_t *cpu_set)
{
  thread_port_t mach_thread;
  int core = 0;

  for (core = 0; core < 8 * cpu_size; core++) {
    if (CPU_ISSET(core, cpu_set)) break;
  }
  thread_affinity_policy_data_t policy = { core };
  mach_thread = pthread_mach_thread_np(thread);
  thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY,
                    (thread_policy_t)&policy, 1);
  return 0;
}
#else
#include <malloc.h>
#endif

#ifdef __clang__
#define inline static inline
#endif

#include <unistd.h>
#include <mpi.h>

#include "../src/common.h"

#define MAXGROUPS 65536		//number of nodes (core processes form a group on a same node)

#ifndef AGGR
#define AGGR (1024*32) //aggregation buffer size per dest in bytes : internode
#endif

#ifndef AGGR_intra
#define AGGR_intra (1024*32) //aggregation buffer size per dest in bytes : intranode
#endif

#define NRECV 4 // number of preposted recvs internode
#define NRECV_intra 4 // number of preposted recvs intranode
#define NSEND 4 // number of available sends internode
#define NSEND_intra 4 // number of send intranode
#define SOATTR __attribute__((visibility("default")))

// Custom timestamp at the beginning of each per-node buffer
// sizeof(double) !!! this may cause cast issues
#define TIMESTAMP_SIZE 8
#define SENDSOURCE(node) (sendbuf + (AGGR + TIMESTAMP_SIZE) * nbuf[node] + TIMESTAMP_SIZE)
#define SENDSOURCE_intra(node) (sendbuf_intra+ ((AGGR_intra + TIMESTAMP_SIZE)*nbuf_intra[node]) + TIMESTAMP_SIZE)

#define ushort unsigned short
static int myproc,num_procs;
static int mygroup,num_groups;
static int mylocal,group_size;
#ifndef PROCS_PER_NODE_NOT_POWER_OF_TWO
static int loggroup,groupmask;
#define PROC_FROM_GROUPLOCAL(g,l) ((l)+((g)<<loggroup))
#define GROUP_FROM_PROC(p) ((p) >> loggroup)
#define LOCAL_FROM_PROC(p) ((p) & groupmask)
#else
#define PROC_FROM_GROUPLOCAL(g,l) ((g)*group_size+(l))
#define GROUP_FROM_PROC(p) ((p)/group_size)
#define LOCAL_FROM_PROC(p) ((p)%group_size)
#endif
volatile static int ack=0;

volatile static int inbarrier=0;

static void (*aml_handlers[256]) (int,void *,int); //pointers to user-provided AM handlers

// extern CustomCommStats *bfs_custom_comm_stats;
extern CustomPacketStats* bfs_custom_packet_stats;
extern uint32_t bfs_custom_packet_stats_i;
extern uint32_t bfs_custom_packet_stats_run_partial_i;
extern int run_number;
extern double *clock_offsets;

//internode comm (proc number X from each group)
//intranode comm (all cores of one nodegroup)
MPI_Comm comm, comm_intra;

// MPI stuff for sends
static char *sendbuf; //coalescing buffers, most of memory is allocated is here
static int *sendsize; //buffer occupacy in bytes
static ushort *acks; //aggregated acks
static ushort *nbuf; //actual buffer for each group/localcore
static ushort activebuf[NSEND];// N_buffer used in transfer(0..NSEND{_intra}-1)
static MPI_Request rqsend[NSEND];
// MPI stuff for recv
static char recvbuf[(AGGR + TIMESTAMP_SIZE)*NRECV];
static MPI_Request rqrecv[NRECV];

unsigned long long nbytes_sent,nbytes_rcvd;

static char *sendbuf_intra;
static int *sendsize_intra;
static ushort *acks_intra;
static ushort *nbuf_intra;
static ushort activebuf_intra[NSEND_intra];
static MPI_Request rqsend_intra[NSEND_intra];
static char recvbuf_intra[(AGGR_intra + TIMESTAMP_SIZE) * NRECV_intra];
static MPI_Request rqrecv_intra[NRECV_intra];
volatile static int ack_intra=0;
inline void aml_send_intra(void *srcaddr, int type, int length, int local ,int from);

void aml_finalize(void);
void aml_barrier(void);

// Custom
void calibrate_clocks() {
    const int SAMPLES = 100;
    double local_times[SAMPLES];
    double remote_times[SAMPLES];
    
    clock_offsets = malloc(num_procs * sizeof(double));
    
    if (myproc == 0) {
        // Root process
        for (int pe = 1; pe < num_procs; pe++) {
            double min_rtt = 1e9;
            double best_offset = 0;
            
            for (int i = 0; i < SAMPLES; i++) {
                double t0 = MPI_Wtime();
                MPI_Send(&t0, 1, MPI_DOUBLE, pe, 0, MPI_COMM_WORLD);
                
                double remote_time;
                MPI_Recv(&remote_time, 1, MPI_DOUBLE, pe, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                double t1 = MPI_Wtime();
                double rtt = t1 - t0;
                
                if (rtt < min_rtt) {
                    min_rtt = rtt;
                    // Assume symmetric network delay
                    best_offset = remote_time - (t0 + rtt/2);
                }
            }
            clock_offsets[pe] = best_offset;
        }
        clock_offsets[0] = 0.0; // Root has no offset
        
    } else {
        // Other processes
        for (int i = 0; i < SAMPLES; i++) {
            double root_time;
            MPI_Recv(&root_time, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            double my_time = MPI_Wtime();
            MPI_Send(&my_time, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }
    
    // Broadcast offsets to all processes
    MPI_Bcast(clock_offsets, num_procs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

SOATTR void aml_register_handler(void(*f)(int,void*,int),int n) { aml_barrier(); aml_handlers[n]=f; aml_barrier(); }

struct __attribute__((__packed__)) hdr { //header of internode message
	ushort sz;
	char hndl;
	char routing;
};
//process internode messages
static void process(int fromgroup,int length ,char* message) {
	// Extract timestamp from the buffer beginning
	double send_timestamp;
	memcpy(&send_timestamp, message, sizeof(double));

	// Record bandwidth measurement
	int full = (((int)AGGR) - length) <= (int)10; // This accounts for the timestamp in the buffers (8B)
	uint32_t next_run_stats_begin = (run_number+1) * MAX_CUSTOM_PACKET_STATS_PER_RUN;
	#ifdef VERBOSE_PRINTS
		if (!full) fprintf(stderr, "size diff: %d-%d-%d=%d  full: %s  --- run: %d full_i=%d, partial_i=%d\n",
			AGGR, sizeof(double),length, AGGR - sizeof(double) - length, full?"Y":"N",
			run_number, bfs_custom_packet_stats_i, bfs_custom_packet_stats_run_partial_i);
		assert(bfs_custom_packet_stats_run_partial_i <= (num_procs*ESTIMATED_MAX_RUN_FRONTIERS));
	#endif
	// Last n cells in the run slice are reserved for partially filled buffers
	if ((!full && bfs_custom_packet_stats_run_partial_i <= (num_procs * ESTIMATED_MAX_RUN_FRONTIERS)) || (full && bfs_custom_packet_stats_i < next_run_stats_begin - (num_procs * ESTIMATED_MAX_RUN_FRONTIERS))) {
		uint32_t stats_idx = bfs_custom_packet_stats_i;
		if (full) {
			bfs_custom_packet_stats_i++;
		} else {
			stats_idx = next_run_stats_begin-(++bfs_custom_packet_stats_run_partial_i);
		}
		double adjusted_recv_time = MPI_Wtime() - clock_offsets[myproc];
		#ifdef VERBOSE_PRINTS
			if (!full) fprintf(stderr, "Registering partially filled packet from %d to %d. stats_idx=%d, comm_time=%f, length - TIMESTAMP_SIZE: %d %d\n", fromgroup, myproc, stats_idx, adjusted_recv_time - send_timestamp, length, TIMESTAMP_SIZE);
		#endif
		bfs_custom_packet_stats[stats_idx].comm_time = adjusted_recv_time - send_timestamp;
		bfs_custom_packet_stats[stats_idx].source = PROC_FROM_GROUPLOCAL(fromgroup,mylocal);
		bfs_custom_packet_stats[stats_idx].destination = myproc;
		bfs_custom_packet_stats[stats_idx].buffer_size = length - TIMESTAMP_SIZE;
	}

	int i = TIMESTAMP_SIZE;
	int from = PROC_FROM_GROUPLOCAL(fromgroup,mylocal);
	while ( i < length ) {
		void* m = message+i;
		struct hdr *h = m;
		int hsz=h->sz;
		int hndl=h->hndl;
		int destlocal = LOCAL_FROM_PROC(h->routing);
		// Register received message (update frontier)
		if(destlocal == mylocal)
			aml_handlers[hndl](from,m+sizeof(struct hdr),hsz);
		else
			aml_send_intra(m+sizeof(struct hdr),hndl,hsz,destlocal,from);
		i += hsz + sizeof(struct hdr);

	}
}
struct __attribute__((__packed__)) hdri { //header of internode message
	ushort routing;
	ushort sz;
	char hndl;
};

//process intranode messages
static void process_intra(int fromlocal,int length ,char* message) {
	int i = TIMESTAMP_SIZE;
	while ( i < length ) {
		void*m = message+i;
		struct hdri *h = m;
		int hsz=h->sz;
		int hndl=h->hndl;
		aml_handlers[hndl](PROC_FROM_GROUPLOCAL((int)(h->routing),fromlocal),m+sizeof(struct hdri),hsz);
		i += sizeof(struct hdri) + hsz;
	}
}

// poll intranode message
inline void aml_poll_intra(void) {
	int flag, from, length,index;
	MPI_Status status;
	MPI_Testany( NRECV_intra,rqrecv_intra, &index, &flag, &status );
	if ( flag ) {
		MPI_Get_count( &status, MPI_CHAR, &length );
		ack_intra -= status.MPI_TAG;
		if(length>0) { //no confirmation & processing for ack only messages
			from = status.MPI_SOURCE;
			if(inbarrier)
				MPI_Send(NULL, 0, MPI_CHAR,from, 1, comm_intra); //ack now
			else
				acks_intra[from]++; //normally we have delayed ack
			process_intra(from, length, recvbuf_intra + (AGGR_intra + TIMESTAMP_SIZE)*index);
		}
		MPI_Start( rqrecv_intra+index);
	}
}
// poll internode message
static void aml_poll(void) {
	int flag, from, length,index;
	MPI_Status status;

	aml_poll_intra();

	MPI_Testany( NRECV,rqrecv,&index, &flag, &status );
	if ( flag ) {
		MPI_Get_count( &status, MPI_CHAR, &length );
		ack -= status.MPI_TAG;
		nbytes_rcvd+=length;
		if(length>0) { //no confirmation & processing for ack only messages
			from = status.MPI_SOURCE;
			if(inbarrier)
				MPI_Send(NULL, 0, MPI_CHAR,from, 1, comm); //ack now
			else
				acks[from]++; //normally we have delayed ack
			process(from, length, recvbuf + (AGGR + TIMESTAMP_SIZE)*index);
		}
		MPI_Start( rqrecv+index );
	}
}

//flush internode buffer to destination node
inline void flush_buffer( int node ) {
	MPI_Status stsend;
	int flag=0,index,tmp;
	if (sendsize[node] == 0 && acks[node]==0 ) return;
	while (!flag) {
		aml_poll();
		MPI_Testany(NSEND,rqsend,&index,&flag,&stsend);
	}

	// Custom stats
	// ++(bfs_custom_comm_stats[CUSTOM_COMM_STATS_IDX(rank, node)].n_comms);
	// bfs_custom_comm_stats[CUSTOM_COMM_STATS_IDX(rank, node)].comms_volume += sendsize[node];

	// Get buffer start (includes reserved timestamp space)
	char* buffer_with_timestamp = sendbuf + (AGGR + TIMESTAMP_SIZE) * nbuf[node];
	double send_time = MPI_Wtime() - clock_offsets[myproc];
	memcpy(buffer_with_timestamp, &send_time, sizeof(double));

	MPI_Isend(buffer_with_timestamp, sendsize[node] + TIMESTAMP_SIZE, MPI_CHAR,node, acks[node], comm, rqsend+index );

	nbytes_sent+=sendsize[node] + TIMESTAMP_SIZE;
	if (sendsize[node] > 0) ack++;
	sendsize[node] = 0;
	acks[node] = 0;
	tmp=activebuf[index]; activebuf[index]=nbuf[node]; nbuf[node]=tmp; //swap bufs
}
//flush intranode buffer, NB:node is local number of pe in group
inline void flush_buffer_intra( int node ) {
	MPI_Status stsend;
	int flag=0,index,tmp;
	if (sendsize_intra[node] == 0 && acks_intra[node]==0 ) return;
	while (!flag) {
		aml_poll_intra();
		MPI_Testany(NSEND_intra,rqsend_intra,&index,&flag,&stsend);
	}

	// Custom stats
	// ++(bfs_custom_comm_stats[CUSTOM_COMM_STATS_IDX(rank, node)].n_comms);
	// bfs_custom_comm_stats[CUSTOM_COMM_STATS_IDX(rank, node)].comms_volume += sendsize_intra[node];

	MPI_Isend( SENDSOURCE_intra(node), sendsize_intra[node], MPI_CHAR,
			node, acks_intra[node], comm_intra, rqsend_intra+index );
	if (sendsize_intra[node] > 0) ack_intra++;
	sendsize_intra[node] = 0;
	acks_intra[node] = 0;
	tmp=activebuf_intra[index]; activebuf_intra[index]=nbuf_intra[node]; nbuf_intra[node]=tmp; //swap bufs

}

inline void aml_send_intra(void *src, int type, int length, int local, int from) {
	//send to _another_ process from same group
	int nmax = AGGR_intra - sendsize_intra[local] - sizeof(struct hdri);
	if ( nmax < length ) {
		flush_buffer_intra(local);
	}
	char* dst = (SENDSOURCE_intra(local)+sendsize_intra[local]);
	struct hdri *h=(void*)dst;
	h->routing = GROUP_FROM_PROC(from);
	h->sz=length;
	h->hndl = type;
	sendsize_intra[local] += length+sizeof(struct hdri);

	memcpy(dst+sizeof(struct hdri),src,length);
}

SOATTR void aml_send(void *src, int type,int length, int node ) {
	if ( node == myproc )
		return aml_handlers[type](myproc,src,length);

	int group = GROUP_FROM_PROC(node);
	int local = LOCAL_FROM_PROC(node);

	//send to another node in my group
	if ( group == mygroup )
		return aml_send_intra(src,type,length,local,myproc);

	//send to another group
	int nmax = AGGR - (sendsize[group] + sizeof(struct hdr));
	if ( nmax < length ) {
		// printf("flushing with %d-%d=%d len %d\n", AGGR, sendsize[group] + sizeof(struct hdr), nmax, length);
		flush_buffer(group);
	}
	char* dst = (SENDSOURCE(group)+sendsize[group]);
	struct hdr *h=(void*)dst;
	h->routing = local;
	h->hndl = type;
	h->sz=length;
	sendsize[group] += length+sizeof(struct hdr);
	memcpy(dst+sizeof(struct hdr),src,length);
}

int stringCmp( const void *a, const void *b)
{ return strcmp(a,b);  }

// Should be called by user instead of MPI_Init()
SOATTR int aml_init( int *argc, char ***argv ) {
	int r, i, j,tmpmax;

	r = MPI_Init(argc, argv);
	CCUTILS_MPI_INIT
	if ( r != MPI_SUCCESS ) return r;

	MPI_Comm_size( MPI_COMM_WORLD, &num_procs );
	MPI_Comm_rank( MPI_COMM_WORLD, &myproc );

	//split communicator
	char host_name[MPI_MAX_PROCESSOR_NAME];
	char (*host_names)[MPI_MAX_PROCESSOR_NAME];
	int namelen,bytes,n,color;
	MPI_Get_processor_name(host_name,&namelen);
	#ifdef FORCE_INTERNODE_ONLY
		color = myproc;
		sprintf(host_name + namelen, "_%d", myproc);
		namelen = strlen(host_name);
	#endif
	CCUTILS_MPI_SECTION_DEF(node_names, "Processes node names")
	CCUTILS_MPI_ALL_PRINT_NAMED(node_names, fprintf(fp, "%s\n", host_name);)
	CCUTILS_MPI_SECTION_END(node_names)

	bytes = num_procs * sizeof(char[MPI_MAX_PROCESSOR_NAME]);
	host_names = (char (*)[MPI_MAX_PROCESSOR_NAME]) malloc(bytes);
	strcpy(host_names[myproc], host_name);
	for (n=0; n<num_procs; n++)
		MPI_Bcast(&(host_names[n]),MPI_MAX_PROCESSOR_NAME, MPI_CHAR, n, MPI_COMM_WORLD);
	qsort(host_names, num_procs, sizeof(char[MPI_MAX_PROCESSOR_NAME]), stringCmp);
	#ifndef FORCE_INTERNODE_ONLY
		color = 0;
		for (n=0; n<num_procs; n++)  {
			if(n>0 && strcmp(host_names[n-1], host_names[n])) color++;
			if(strcmp(host_name, host_names[n]) == 0) break;
		}
	#endif
	free(host_names);
	MPI_Comm_split(MPI_COMM_WORLD, color, myproc, &comm_intra);

	//find intranode numbers and make internode communicator
	MPI_Comm_size( comm_intra, &group_size );
	MPI_Comm_rank( comm_intra, &mylocal );

	MPI_Comm_split(MPI_COMM_WORLD, mylocal, myproc, &comm);

	MPI_Comm_size( comm, &num_groups );
	MPI_Comm_rank( comm, &mygroup );

	//first nonblocking barriers are blocking,so we call them now
	MPI_Request hndl;
	MPI_Ibarrier(comm,&hndl);
	MPI_Wait(&hndl,MPI_STATUS_IGNORE);
	MPI_Ibarrier(comm_intra,&hndl);
	MPI_Wait(&hndl,MPI_STATUS_IGNORE);

#ifndef PROCS_PER_NODE_NOT_POWER_OF_TWO
	groupmask=group_size-1;
	if((group_size&groupmask)) { printf("AML: Fatal: non power2 groupsize unsupported. Define macro PROCS_PER_NODE_NOT_POWER_OF_TWO to override\n");return -1;}
	for (loggroup = 0; loggroup < group_size; loggroup++)
		if ((1 << loggroup) == group_size) break;
#endif
	if(myproc!=PROC_FROM_GROUPLOCAL(mygroup,mylocal)) {printf("AML: Fatal: Strange group rank assignment scheme.\n");return -1;}
	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);

	CPU_SET(mylocal,&cpuset); //FIXME ? would it work good enough on all architectures?
	pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

	CCUTILS_MPI_SECTION_DEF(bfs_config, "BFS Communication and Local Configuration")
	CCUTILS_MPI_PRINTF_ONCE("multicore=true\nnum_groups=%d\ngroup_size=%d\n", num_groups, group_size);
	CCUTILS_MPI_PRINTF_ONCE("loggroup=%d\ngroupmask=%d\n", loggroup, groupmask);
	CCUTILS_MPI_PRINTF_ONCE("NRECV=%d\nNRECVi=%d\nNSEND=%d\nNSENDi=%d\nAGGR=%dK\nAGGRi=%dK\n",NRECV,NRECV_intra,NSEND,NSEND_intra,AGGR>>10,AGGR_intra>>10);
	#ifdef PROCS_PER_NODE_NOT_POWER_OF_TWO
		CCUTILS_MPI_PRINTF_ONCE("PROCS_PER_NODE_POWER_OF_TWO=false\n");
	#else
		CCUTILS_MPI_PRINTF_ONCE("PROCS_PER_NODE_POWER_OF_TWO=true\n");
	#endif
	CCUTILS_MPI_SECTION_END(bfs_config)
	if(num_groups>MAXGROUPS) { if(myproc==0) printf("AML:v1.0 reference:unsupported num_groups > MAXGROUPS=%d\n",MAXGROUPS); exit(-1); }
	fflush(NULL);
	//init preposted recvs: NRECV internode
	for(i=0;i<NRECV;i++)  {
		r = MPI_Recv_init(  recvbuf + (AGGR + TIMESTAMP_SIZE)*i, 
							AGGR + TIMESTAMP_SIZE, 
							MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, 
							comm, rqrecv+i);
    	if (r != MPI_SUCCESS) return r;
	}
	// Allocate extra space for timestamps at the start of each buffer
	sendbuf = malloc((AGGR + TIMESTAMP_SIZE) * (num_groups + NSEND));
	if (!sendbuf) return -1;
	memset(sendbuf, 0, (AGGR + TIMESTAMP_SIZE) * (num_groups + NSEND));
	sendsize = malloc( num_groups*sizeof(*sendsize) );
	if (!sendsize) return -1;
	acks = malloc( num_groups*sizeof(*acks) );
	if (!acks) return -1;
	nbuf = malloc( num_groups*sizeof(*nbuf) );
	if (!nbuf) return -1;


	for(i=0; i<NRECV_intra; i++) {
		r = MPI_Recv_init(  recvbuf_intra + (AGGR_intra + TIMESTAMP_SIZE)*i, 
							AGGR_intra + TIMESTAMP_SIZE, 
							MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, 
							comm_intra, rqrecv_intra+i);
		if (r != MPI_SUCCESS) return r;
	}
	sendbuf_intra = malloc( AGGR_intra*(group_size+NSEND_intra));
	if ( !sendbuf_intra ) return -1;
	memset(sendbuf_intra,0,AGGR_intra*(group_size+NSEND_intra));
	sendsize_intra = malloc( group_size*sizeof(*sendsize_intra) );
	if (!sendsize_intra) return -1;
	acks_intra = malloc( group_size*sizeof(*acks_intra) );
	if (!acks_intra) return -1;
	nbuf_intra = malloc( group_size*sizeof(*nbuf_intra) );
	if (!nbuf_intra) return -1;
	for ( j = 0; j < group_size; j++ ) {
		sendsize_intra[j] = 0; nbuf_intra[j] = j; acks_intra[j]=0;
	}
	for(i=0;i<NRECV_intra;i++)
		MPI_Start(rqrecv_intra+i);

	for ( j = 0; j < NSEND_intra; j++ ) {
		MPI_Isend( NULL, 0, MPI_CHAR, MPI_PROC_NULL, 0, comm_intra, rqsend_intra+j );
		activebuf_intra[j]=group_size+j;
	}

	for ( j = 0; j < num_groups; j++ ) {
		sendsize[j] = 0; nbuf[j] = j;  acks[j]=0;
	}
	for(i=0;i<NRECV;i++)
		MPI_Start( rqrecv+i );
	for ( j = 0; j < NSEND; j++ ) {
		MPI_Isend( NULL, 0, MPI_CHAR, MPI_PROC_NULL, 0, comm, rqsend+j );
		activebuf[j]=num_groups+j;
	}

	// Custom
	calibrate_clocks();
	return 0;
}

SOATTR void aml_barrier( void ) {
	int i,flag;
	MPI_Request hndl;
	inbarrier++;
	//1. flush internode buffers
	for ( i = 1; i < num_groups; i++ ) {
		int group=(mygroup+i)%num_groups;
		flush_buffer(group);
	}
	//2. wait for all internode being acknowledged
	while(ack!=0) aml_poll();
	//3. notify everybody that all my internode messages were received
	MPI_Ibarrier(comm,&hndl);
	//4. receive internode until barrier done
	flag=0;
	while(flag==0) {
		MPI_Test(&hndl,&flag,MPI_STATUS_IGNORE); aml_poll(); }
	// NB: All internode received here. I can receive some more intranode.

	//5. Flush all intranode buffers
	for ( i = 1; i < group_size; i++ ) {
		int localproc=LOCAL_FROM_PROC(mylocal+i);
		flush_buffer_intra(localproc);
	}
	//inbarrier=2;
	//6. wait for all intranode being acknowledged
	while(ack_intra!=0) aml_poll_intra();
	//7. notify everybody that all my intranode messages were received
	MPI_Ibarrier(comm_intra,&hndl);
	//8. receive internode until barrier done
	flag=0;
	while(flag==0) {
		MPI_Test(&hndl,&flag,MPI_STATUS_IGNORE); aml_poll_intra(); }
	inbarrier--;
	MPI_Barrier(MPI_COMM_WORLD);
}

SOATTR void aml_finalize( void ) {
	int i;
	aml_barrier();
	for(i=0;i<NRECV;i++)
		MPI_Cancel(rqrecv+i);
#ifndef NOINTRA
	for(i=0;i<NRECV_intra;i++)
		MPI_Cancel(rqrecv_intra+i);
	MPI_Status stat_intra[NSEND_intra];
	MPI_Waitall(NSEND_intra,rqsend_intra,stat_intra);
#endif
	MPI_Status stat[NSEND];
	MPI_Waitall(NSEND,rqsend,stat);
	MPI_Finalize();
}

SOATTR int aml_my_pe(void) { return myproc; }
SOATTR int aml_n_pes(void) { return num_procs; }
