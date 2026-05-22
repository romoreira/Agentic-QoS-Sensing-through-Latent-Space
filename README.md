# BESS UPF QoS and Slice Priority Notes

## Purpose

This README documents what was learned while debugging and analyzing the current BESS UPF deployment used in the `Agentic-B5G-QoS-Enforcement` environment.

The practical goal is not only to create PFCP sessions and enforce per session throughput limits. The broader goal is to understand whether the current BESS UPF can support a stronger policy in which one slice is favored over another, for example Gold traffic being protected while Bronze traffic is degraded first during overload.

This document is written for someone who has not followed the debugging session and needs to understand the deployment, what was confirmed, what the current UPF can already do, what it cannot do directly, and what would be required to implement slice aware prioritization.

## High level conclusion

The current BESS UPF pipeline already supports PFCP installed PDR, FAR, and QER rules. It also supports per session and application level rate limiting through QERs, which can be modified at runtime using `pfcpctl session modify`.

However, the current pipeline does not implement strict or weighted priority between slices out of the box. It performs metering and policing, not slice aware queue scheduling.

To implement real priority between slices, such as Gold being served before Bronze when they compete for the same N6 output path, the BESS pipeline must be modified. PFCP alone is not enough for queue priority. PFCP can provide classification information and rate limits, but BESS must execute the actual datapath scheduling.

## Deployment context

The deployment uses BESS UPF with DPDK and Mellanox interfaces.

The working deployment observed during the session used these logical roles.

```text
N3 access interface
enp7s0np0 or enp8s0np0 depending on the deployment generation

N6 core interface
enp8s0np0 or enp9s0np0 depending on the deployment generation

PFCP control interface
enp7s0 or enp9s0 depending on the deployment generation
```

One important lesson is that interface names must be interpreted according to the specific startup script used. An older deployment used one mapping, while the newer filtered Infiniband startup script used another mapping.

The script `patch_upf_filtered_infiniband_start.sh` was especially important because it filters `/dev/infiniband` devices exposed to the BESS container. This avoids a control VF being detected first by DPDK and shifting the PMD port order. When this script works correctly, BESS sees only the two datapath NICs, not the PFCP control NIC.

The startup script validates the following items.

```text
BESS recognizes exactly two DPDK PMD ports
BESS gRPC listens on port 10514
pfcpiface listens on UDP port 8805
BESS datapath port MACs match the expected N3 and N6 NICs
N6 neighbor and route entries are installed
PDR and FAR tables are empty before pfcpsim creates sessions
```

Before creating sessions, zero PDR and FAR rules is expected.

## Control plane flow

The control plane path uses `pfcpsim` and `pfcpctl` from the TRex host.

The correct order is always this.

```bash
sudo docker rm -f pfcpsim 2>/dev/null || true

sudo docker run --rm -d --network host \
  --name pfcpsim \
  pfcpsim:patched \
  -p 12345 \
  --interface enp9s0

sudo docker exec pfcpsim pfcpctl -s localhost:12345 service configure \
  --n3-addr 192.168.70.1 \
  --remote-peer-addr 192.168.90.1

sudo docker exec pfcpsim pfcpctl -s localhost:12345 service associate

sudo docker exec pfcpsim pfcpctl -s localhost:12345 session create \
  --count 3 --baseID 1 \
  --ue-pool 10.250.0.0/24 \
  --gnb-addr 192.168.70.2 \
  --app-filter "udp:any:any:allow:100" \
  --session-mbr-uplink   1000000,1000000,1000000 \
  --session-mbr-downlink 1000000,1000000,1000000 \
  --app-mbr-uplink       50000,200000,500000 \
  --app-mbr-downlink     50000,200000,500000
```

After session creation, the UPF should show installed rules in the BESS modules.

```bash
sudo docker exec bess /opt/bess/bessctl/bessctl show module pdrLookup | grep rules
sudo docker exec bess /opt/bess/bessctl/bessctl show module farLookup | grep rules
sudo docker exec bess /opt/bess/bessctl/bessctl show module appQERLookup | grep rules
sudo docker exec bess /opt/bess/bessctl/bessctl show module sessionQERLookup | grep rules
```

A working example showed the following rule counts.

```text
pdrLookup
6 rules

farLookup
6 rules

appQERLookup
12 rules

sessionQERLookup
6 rules
```

These numbers make sense for three sessions because the UPF installs uplink and downlink rules and multiple QER entries.

## What was confirmed in the working state

The BESS pipeline was confirmed to have the following relevant modules.

```text
pdrLookup
gtpuDecap
appQERLookup
sessionQERLookup
farLookup
farMerge
sliceMeter
executeFAR
N6 route module
```

The working pipeline showed heavy packet counters in the main uplink path.

For example, `pdrLookup` showed six rules and traffic leaving through the GTP-U decapsulation gate.

```text
pdrLookup
6 rules

gate 1 toward gtpuDecap
large packet count
```

The app level QER module showed twelve rules and three main output behaviors.

```text
appQERLookup green or yellow
continues to sessionQERLookup

appQERLookup red
goes to appQERMeterRed

appQERLookup fail or status drop
goes to the corresponding sink
```

The session level QER module showed six rules and similar behavior.

```text
sessionQERLookup green or yellow
continues to farLookup

sessionQERLookup red
goes to sessionQERMeterRed
```

The FAR lookup module showed six rules and forwarded traffic toward `farMerge`.

```text
farLookup
6 rules

gate 0
forward action toward farMerge
```

These observations confirm that the BESS UPF datapath is active and not merely configured.

## What PFCP and QER already provide

PFCP with PDR, FAR, and QER already provides a working mechanism for rate limiting.

The relevant elements are.

```text
PDR
identifies which packet or flow matches a rule

FAR
defines forwarding behavior

QER
defines QoS enforcement, including MBR based policing
```

The `pfcpsim` image supports runtime session modification.

```bash
sudo docker exec pfcpsim pfcpctl -s localhost:12345 session modify --help
```

The available options include.

```text
session-mbr-uplink
session-mbr-downlink
app-mbr-uplink
app-mbr-downlink
qfi
app-filter
```

A test command was successfully accepted by `pfcpsim`.

```bash
sudo docker exec pfcpsim pfcpctl -s localhost:12345 session modify \
  --count 1 --baseID 1 \
  --ue-pool 10.250.0.0/24 \
  --gnb-addr 192.168.70.2 \
  --app-filter "udp:any:any:allow:100" \
  --session-mbr-uplink   1000000 \
  --session-mbr-downlink 1000000 \
  --app-mbr-uplink       10000 \
  --app-mbr-downlink     10000
```

This modifies one session and changes the application level MBR to 10 Mbps because the value is expressed in kbps.

This is the correct way to modify throughput limits without desynchronizing `pfcpiface` and the BESS datapath.

## What this does not provide

The QER based path does not create true priority between slices.

It can implement this policy.

```text
Gold keeps a high MBR
Silver is reduced moderately
Bronze is reduced aggressively
```

But it does not implement this policy.

```text
When Gold and Bronze compete for the same output path, Gold packets are scheduled first.
```

That second behavior is queue scheduling, not merely rate limiting.

## Rate limiting versus priority

This distinction is central.

### Rate limiting

Rate limiting means each class or session is capped.

Example.

```text
Gold
200 Mbps

Silver
100 Mbps

Bronze
20 Mbps
```

If Bronze exceeds its limit, packets can be dropped or marked red by the QER meter.

This is already available through PFCP and QER.

### Priority scheduling

Priority scheduling means the output link is shared according to class priority.

Example.

```text
Gold
served first

Silver
served after Gold

Bronze
served only when higher priority traffic leaves capacity
```

This requires queues and a scheduler in the datapath.

This is not currently implemented in the default BESS UPF pipeline.

## Current sliceMeter behavior

The module `sliceMeter` exists in the pipeline.

```text
sliceMeter::Qos
2 fields
2 rules
```

However, it reads only these fields.

```text
action
tunnel_out_type
```

This means the current `sliceMeter` does not distinguish slices by.

```text
QFI
F-SEID
QER ID
TEID
UE IP
S-NSSAI
slice name
```

Therefore, the current `sliceMeter` should not be interpreted as Gold, Silver, Bronze slice priority. It appears to enforce a broader meter based on forwarding action and tunnel output type.

## Why ARP and N6 routing are not the current priority mechanism

The startup script installs a static N6 neighbor and route. This is important for sending decapsulated packets toward the N6 receiver.

A typical N6 neighbor and route setup is.

```bash
sudo ip neigh replace 192.168.80.2 lladdr 10:70:fd:c0:ef:40 dev <N6_IF> nud permanent
sudo ip route replace 192.168.80.2/32 via 192.168.80.2 dev <N6_IF> onlink
```

This is necessary for forwarding after FAR execution.

However, ARP and N6 routing do not explain whether slices are prioritized. They only make sure the packet can be sent to the next hop after the forwarding decision is already made.

If `pdrLookup`, `farLookup`, and QER modules have zero rules, ARP is irrelevant. If these modules have rules and traffic reaches `executeFAR`, then N6 routing matters for delivery.

## Why ping is not a reliable datapath validation

In this setup, only the control interface is expected to behave like a normal Linux interface.

```text
PFCP control interface
kernel visible
ping works

N3 and N6 datapath interfaces
controlled by DPDK, BESS, or TRex
ping from Linux may not work
```

Therefore, being able to ping only `192.168.90.1` is normal. This validates the PFCP control path, not the datapath.

The correct datapath validation uses BESS counters.

```bash
sudo docker exec bess /opt/bess/bessctl/bessctl show port
sudo docker exec bess /opt/bess/bessctl/bessctl show module pdrLookup
sudo docker exec bess /opt/bess/bessctl/bessctl show module gtpuDecap
sudo docker exec bess /opt/bess/bessctl/bessctl show module appQERLookup
sudo docker exec bess /opt/bess/bessctl/bessctl show module sessionQERLookup
sudo docker exec bess /opt/bess/bessctl/bessctl show module farLookup
```

## What BESS supports for priority

The BESS image includes several modules and scheduling primitives relevant to priority.

```text
Qos
Queue
DRR
Split
Merge
RoundRobin
RandomSplit
WorkerSplit
```

The important finding is that BESS traffic classes support these policies.

```text
priority
round_robin
weighted_fair
rate_limit
```

This was confirmed by the examples in the BESS container.

The most relevant example is `qtest.bess`.

```python
bess.add_tc('w2root', policy='priority', wid=1)
bess.add_tc('queue_rr', parent='w2root', policy='round_robin', priority=1)
queue.attach_task(parent='queue_rr')
bess.add_tc('fast', parent='w2root', priority=0, policy='rate_limit', resource='packet', limit={'packet': 100000})
src2.attach_task(parent='fast')
```

This confirms that BESS can represent priority scheduling. In BESS, lower priority numbers indicate higher priority.

```text
priority 0
highest priority

priority 1
lower priority

priority 2
even lower priority
```

## What DRR provides

BESS also includes `DRR`.

```text
DRR
Deficit Round Robin
```

The `DRR` module has these parameters.

```text
num_flows
quantum
max_flow_queue_size
```

The example uses.

```python
q::DRR(num_flows=4, quantum=1000)
```

This suggests the current DRR module provides fairness across flows, but it does not directly expose independent weights per slice such as.

```text
Gold quantum 3000
Silver quantum 1500
Bronze quantum 500
```

Therefore, DRR may be useful for fairness, but the more direct path for Gold, Silver, Bronze priority is BESS traffic classes with `policy='priority'`.

## Where the pipeline should be modified

The main uplink path enters the shared pipeline, passes through PDR, QER, FAR, and then reaches `executeFAR`.

The important portion of `up4.bess` is.

```python
if not parser.measure_flow:
    executeFAR:farForwardUAction \
        -> ports[parser.core_ifname].rtr
else:
    executeFAR:farForwardUAction \
        -> postULQosFlowMeasure::FlowMeasure(leader=False,
                                             flag_attr_name="buffer_flag",
                                             entries=parser.table_size_flow_measure) \
        -> ports[parser.core_ifname].rtr
```

This is the best insertion point for uplink priority toward N6.

The conceptual replacement is.

```text
executeFAR:farForwardUAction
        |
        v
slice classifier
        |
        +--> Gold queue
        |
        +--> Silver queue
        |
        +--> Bronze queue
        |
        v
scheduler or traffic class tree
        |
        v
ports[parser.core_ifname].rtr
```

If `measure_flow` is enabled, priority can be inserted either before or after `postULQosFlowMeasure`, depending on what metric is desired.

If the goal is to measure the effect of priority, placing measurement before and after scheduling may be useful.

## Candidate classification fields

The classifier must decide whether a packet belongs to Gold, Silver, or Bronze.

Possible fields are.

```text
QFI
good semantic match for QoS flow or class
requires PFCP session creation with meaningful qfi values

F-SEID
available in metadata
useful for mapping sessions to classes

QER ID
available before some later modules
good for policy based mapping, but may be overwritten or less safe after FAR processing

UE IP
possible earlier in the pipeline
less direct after decapsulation and FAR processing

TEID
useful on N3 before decapsulation
less useful after executeFAR
```

For this particular pipeline location, after `executeFAR`, `F-SEID` is likely safer than `QER ID` because later modules may reuse metadata offsets. `QFI` could be a clean option if sessions are created with explicit `--qfi` values and the metadata remains available at the insertion point.

## Recommended implementation strategies

### Strategy 1

Use PFCP and QER only.

This is the safest option and works today.

Policy example.

```text
Gold
keep high MBR

Silver
reduce moderately under overload

Bronze
reduce aggressively under overload
```

How to implement.

```bash
sudo docker exec pfcpsim pfcpctl -s localhost:12345 session modify ...
```

Advantages.

```text
No BESS pipeline modification
pfcpiface and BESS remain synchronized
Easy to automate with an agentic controller
Good for an SLA constrained QoS paper
```

Limitations.

```text
Not true priority scheduling
Bronze is sacrificed by rate limit, not by queue priority
```

### Strategy 2

Modify BESS with queue based priority.

This is the option that matches the desired behavior most closely.

Policy example.

```text
Gold traffic enters high priority queue
Silver traffic enters medium priority queue
Bronze traffic enters low priority queue
```

Implementation idea.

```python
bess.add_tc('slice_root', policy='priority', wid=0)

bess.add_tc('gold_tc', parent='slice_root', policy='round_robin', priority=0)
bess.add_tc('silver_tc', parent='slice_root', policy='round_robin', priority=1)
bess.add_tc('bronze_tc', parent='slice_root', policy='round_robin', priority=2)

goldQueue.attach_task(parent='gold_tc')
silverQueue.attach_task(parent='silver_tc')
bronzeQueue.attach_task(parent='bronze_tc')
```

Advantages.

```text
Real priority behavior
Matches the idea of one slice being favored over another
Can be evaluated under overload
```

Limitations.

```text
Requires editing up4.bess
Requires choosing and preserving a slice classification field
Requires careful testing to avoid breaking the UPF pipeline
May require changes to startup scripts so the patch is persistent
```

### Strategy 3

Clean integration with PFCP and BESS.

This is the more complete research grade design.

PFCP or pfcpsim provides the slice class information.

```text
QFI
QER ID
F-SEID to slice class mapping
```

pfcpiface installs or exports the mapping.

BESS uses the mapping to classify packets into queues.

The datapath then performs.

```text
classification
queueing
priority scheduling
forwarding to N6
```

Advantages.

```text
Clean separation between control plane policy and datapath enforcement
More defensible as a B5G slicing architecture
Can support dynamic agentic control
```

Limitations.

```text
Requires modifications in both pfcpiface and up4.bess
More engineering effort
Requires reproducibility tests
```

## Minimal safe experimental path

The safest path is incremental.

### Step 1

Keep the current working PFCP and QER pipeline.

Validate that sessions are installed.

```bash
sudo docker exec bess /opt/bess/bessctl/bessctl show module pdrLookup | grep rules
sudo docker exec bess /opt/bess/bessctl/bessctl show module appQERLookup | grep rules
sudo docker exec bess /opt/bess/bessctl/bessctl show module sessionQERLookup | grep rules
sudo docker exec bess /opt/bess/bessctl/bessctl show module farLookup | grep rules
```

Expected result.

```text
pdrLookup
6 rules

appQERLookup
12 rules

sessionQERLookup
6 rules

farLookup
6 rules
```

### Step 2

Validate that PFCP based MBR modification works.

Modify only one session.

```bash
sudo docker exec pfcpsim pfcpctl -s localhost:12345 session modify \
  --count 1 --baseID 1 \
  --ue-pool 10.250.0.0/24 \
  --gnb-addr 192.168.70.2 \
  --app-filter "udp:any:any:allow:100" \
  --session-mbr-uplink   1000000 \
  --session-mbr-downlink 1000000 \
  --app-mbr-uplink       10000 \
  --app-mbr-downlink     10000
```

Then generate traffic and observe whether `appQERMeterRed` increases.

```bash
sudo docker exec bess /opt/bess/bessctl/bessctl show module appQERLookup | grep -E 'appQERMeterRed|rules'
```

### Step 3

Create a copy of `up4.bess`.

```bash
sudo docker exec bess bash -lc \
'cp /opt/bess/bessctl/conf/up4.bess /opt/bess/bessctl/conf/up4.bess.bak_priority'
```

Do not patch the live pipeline before confirming the minimal Queue insertion works.

### Step 4

Insert a single Queue between `executeFAR:farForwardUAction` and `ports[parser.core_ifname].rtr`.

This tests whether queueing works in the N6 uplink path.

Conceptual change.

```python
executeFAR:farForwardUAction \
    -> ulTestQueue::Queue() \
    -> ports[parser.core_ifname].rtr
```

If `measure_flow` is enabled, the analogous path is.

```python
executeFAR:farForwardUAction \
    -> postULQosFlowMeasure::FlowMeasure(...) \
    -> ulTestQueue::Queue() \
    -> ports[parser.core_ifname].rtr
```

or.

```python
executeFAR:farForwardUAction \
    -> ulTestQueue::Queue() \
    -> postULQosFlowMeasure::FlowMeasure(...) \
    -> ports[parser.core_ifname].rtr
```

The best choice depends on whether measurement should include queueing effects.

### Step 5

If one queue works, add Gold, Silver, and Bronze queues.

Conceptual design.

```text
executeFAR:farForwardUAction
        |
        v
sliceClassify
        |
        +--> goldQueue
        +--> silverQueue
        +--> bronzeQueue
        |
        v
priority traffic class tree
        |
        v
N6 route
```

### Step 6

Attach queues to a priority traffic class tree.

Conceptual BESS code.

```python
bess.add_tc('slice_root', policy='priority', wid=0)

bess.add_tc('gold_tc', parent='slice_root', policy='round_robin', priority=0)
bess.add_tc('silver_tc', parent='slice_root', policy='round_robin', priority=1)
bess.add_tc('bronze_tc', parent='slice_root', policy='round_robin', priority=2)

goldQueue.attach_task(parent='gold_tc')
silverQueue.attach_task(parent='silver_tc')
bronzeQueue.attach_task(parent='bronze_tc')
```

### Step 7

Validate behavior under overload.

Expected behavior.

```text
Gold queue remains served first
Silver degrades after Gold pressure increases
Bronze suffers first under congestion
```

Useful counters.

```bash
sudo docker exec bess /opt/bess/bessctl/bessctl show tc
sudo docker exec bess /opt/bess/bessctl/bessctl show module goldQueue
sudo docker exec bess /opt/bess/bessctl/bessctl show module silverQueue
sudo docker exec bess /opt/bess/bessctl/bessctl show module bronzeQueue
sudo docker exec bess /opt/bess/bessctl/bessctl command module goldQueue get_status QueueCommandGetStatusArg
sudo docker exec bess /opt/bess/bessctl/bessctl command module silverQueue get_status QueueCommandGetStatusArg
sudo docker exec bess /opt/bess/bessctl/bessctl command module bronzeQueue get_status QueueCommandGetStatusArg
```

## Runtime control implications

There are two levels of runtime control.

### Runtime control already available

PFCP based QER and MBR modification.

This can be controlled by an agentic controller today.

```text
observe telemetry
choose new MBR vector
apply pfcpctl session modify
validate BESS counters
```

### Runtime control that requires BESS changes

Changing scheduling behavior among Gold, Silver, and Bronze queues at runtime.

This requires either.

```text
traffic class commands
runtime BESS task reconfiguration
predefined queues with adjustable rate limits
custom BESS module
```

A practical design is to keep the queues fixed and let the agent modify their rate limits or PFCP MBRs. This gives dynamic behavior without rebuilding the BESS graph every control window.

## Suggested research framing

A good research framing is not simply that the UPF can set bitrate. The stronger framing is this.

```text
The baseline BESS UPF provides PFCP based QoS enforcement through QER metering.
Our extension moves from per session rate policing to slice aware datapath prioritization, where the UPF can protect high priority slices under contention by steering traffic through priority aware BESS scheduling paths.
```

This creates a clear distinction.

```text
Baseline
PFCP QER MBR enforcement

Enhanced UPF
slice aware classification plus queue scheduling in BESS

Agentic controller
observes telemetry and decides whether to adjust QER limits, queue priorities, or both
```

## Recommended next experiments

### Experiment 1

Validate PFCP QER modification.

Goal.

```text
Show that app MBR changes affect packet coloring and red meter counters.
```

Expected metric.

```text
appQERMeterRed packets increase when traffic exceeds the modified MBR.
```

### Experiment 2

Validate one inserted Queue.

Goal.

```text
Confirm that packets still flow after adding a Queue before N6 routing.
```

Expected metric.

```text
No loss of forwarding
Queue status visible through get_status
N6 output counter still increases
```

### Experiment 3

Validate three queues without priority.

Goal.

```text
Confirm that slice classification sends packets into Gold, Silver, and Bronze queues.
```

Expected metric.

```text
Each queue receives only the expected class.
```

### Experiment 4

Validate priority traffic classes.

Goal.

```text
Confirm Gold is served before Bronze under output contention.
```

Expected metric.

```text
Gold throughput and latency remain better than Bronze when aggregate demand exceeds N6 capacity.
```

### Experiment 5

Compare policies.

Possible policies.

```text
PFCP QER only
BESS priority only
QER plus BESS priority
Agentic dynamic policy
```

Metrics.

```text
throughput by slice
loss by slice
latency by slice
queue occupancy by slice
red meter packets by slice
policy reaction time
control overhead
```

## Final recommendation

For the current objective, there are two valid tracks.

Use PFCP `session modify` if the goal is adaptive throughput enforcement.

Modify `up4.bess` if the goal is real slice priority.

The most defensible architecture combines both.

```text
PFCP
installs PDR, FAR, QER, QFI, and session metadata

BESS
classifies packets into slice classes and applies queue scheduling

Agentic controller
decides when to preserve, degrade, or restore slice resources based on telemetry
```

This gives a clear technical contribution beyond simply changing MBR values. It turns the UPF into a programmable, slice aware datapath that can enforce differentiated service under contention.
