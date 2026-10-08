/**
 * Flatten ComfyUI subgraphs into a plain editor export.
 *
 * A modern ComfyUI workflow is not a flat list of nodes. An author can collapse a
 * section into a **subgraph** and place one or more instances of it; the file then
 * carries a `definitions.subgraphs` table and a node whose `type` is the
 * subgraph's *UUID* rather than a node class. ComfyUI's editor resolves that at
 * run time, and the API does not: queueing the file as-is asks the engine for a
 * node class called `f2fdebf6-…`, which it has never heard of.
 *
 * This module does that resolution ahead of time, so the rest of the pipeline only
 * ever sees real node classes and never needs an API export from the operator.
 *
 * ## What an instance actually is
 *
 * A subgraph definition is a workflow in miniature with two virtual endpoints:
 *
 * - an **input proxy** (`inputNode`, conventionally id `-10`) whose output slots
 *   are the subgraph's public inputs, in the order `inputs[]` declares them;
 * - an **output proxy** (`outputNode`, conventionally id `-20`) whose input slots
 *   are its public outputs, in the order `outputs[]` declares them.
 *
 * A link inside the definition that originates at the input proxy *is* a reference
 * to a public input; one that terminates at the output proxy *is* the value the
 * instance hands back.
 *
 * Flattening therefore means, for each instance:
 *
 * 1. copy the definition's nodes under a prefix that includes the instance id, so
 *    two instances of one subgraph cannot collide;
 * 2. replace every input-proxy link with either the link the instance received
 *    from outside, or — when the instance port is unconnected — the **value**
 *    bound to that port, which is the per-instance override;
 * 3. remember what the output proxy was fed, so links leaving the instance can be
 *    re-pointed at the real internal node;
 * 4. recurse, because a definition may itself place instances of another subgraph.
 *
 * ## Per-instance overrides
 *
 * ComfyUI writes an instance's settings twice: positionally in `widgets_values`
 * (in the declaration order of `inputs[]`) and, in newer revisions, by name in
 * `widgets_values_named`. The named form is authoritative when present, because a
 * positional array silently mis-binds the moment an author reorders a port. Both
 * are read, and the positional form is the fallback.
 *
 * @module dsh-ai-model-hub/discovery/comfyui-subgraph
 */
import { isRecordLike, readArray, readString } from "./http.js";
/**
 * Read one link in either serialization.
 *
 * The editor writes top-level links as positional arrays
 * (`[id, origin, slot, target, slot, type]`) and links *inside a subgraph
 * definition* as objects keyed by name. Both appear in the same file, so both are
 * accepted rather than assuming one shape and silently dropping the other.
 *
 * @param entry - the raw link.
 * @returns the normalized link, or `undefined` when it is not a link.
 */
export function readUiLink(entry) {
    if (Array.isArray(entry)) {
        const [id, originId, originSlot, targetId, targetSlot, type] = entry;
        if (typeof id !== 'number')
            return undefined;
        const origin = typeof originId === 'string' || typeof originId === 'number' ? String(originId) : undefined;
        const target = typeof targetId === 'string' || typeof targetId === 'number' ? String(targetId) : undefined;
        if (origin === undefined || target === undefined)
            return undefined;
        return {
            id,
            originId: origin,
            originSlot: typeof originSlot === 'number' ? originSlot : 0,
            targetId: target,
            targetSlot: typeof targetSlot === 'number' ? targetSlot : 0,
            type: typeof type === 'string' ? type : '',
        };
    }
    if (isRecordLike(entry)) {
        const id = entry['id'];
        const originId = entry['origin_id'];
        const targetId = entry['target_id'];
        if (typeof id !== 'number')
            return undefined;
        const origin = typeof originId === 'string' || typeof originId === 'number' ? String(originId) : undefined;
        const target = typeof targetId === 'string' || typeof targetId === 'number' ? String(targetId) : undefined;
        if (origin === undefined || target === undefined)
            return undefined;
        const originSlot = entry['origin_slot'];
        const targetSlot = entry['target_slot'];
        const type = entry['type'];
        return {
            id,
            originId: origin,
            originSlot: typeof originSlot === 'number' ? originSlot : 0,
            targetId: target,
            targetSlot: typeof targetSlot === 'number' ? targetSlot : 0,
            type: typeof type === 'string' ? type : '',
        };
    }
    return undefined;
}
/**
 * Read the subgraph definitions of a document, keyed by the id a node references.
 *
 * @param document - the UI-format document.
 * @returns the definitions, keyed by subgraph id.
 */
export function readSubgraphDefinitions(document) {
    const definitions = new Map();
    const container = document['definitions'];
    if (!isRecordLike(container))
        return definitions;
    for (const raw of readArray(container, 'subgraphs')) {
        if (!isRecordLike(raw))
            continue;
        const id = readString(raw, 'id');
        if (id === undefined)
            continue;
        const inputs = readArray(raw, 'inputs');
        const inputNames = [];
        const inputTypes = [];
        for (const entry of inputs) {
            if (!isRecordLike(entry))
                continue;
            inputNames.push(readString(entry, 'name') ?? '');
            inputTypes.push(readString(entry, 'type') ?? '');
        }
        const links = readArray(raw, 'links')
            .map((entry) => readUiLink(entry))
            .filter((link) => link !== undefined);
        const definition = {
            id,
            name: readString(raw, 'name') ?? id,
            // The proxies are named rather than assumed, so a revision that renumbers
            // them does not silently produce a graph wired to a node that is not there.
            inputNodeId: proxyId(raw['inputNode'], '-10'),
            outputNodeId: proxyId(raw['outputNode'], '-20'),
            inputNames,
            inputTypes,
            nodes: readArray(raw, 'nodes').filter(isRecordLike),
            links,
        };
        definitions.set(id, definition);
    }
    return definitions;
}
/**
 * The id of a proxy endpoint, which the editor may write as an id or an object.
 * @param raw - the proxy value.
 * @param fallback - the conventional id.
 * @returns the id.
 */
function proxyId(raw, fallback) {
    if (typeof raw === 'string' || typeof raw === 'number')
        return String(raw);
    if (isRecordLike(raw)) {
        const id = raw['id'];
        if (typeof id === 'string' || typeof id === 'number')
            return String(id);
    }
    return fallback;
}
/** Allocate a link id this pass has not used. */
function allocateLink(context) {
    const id = context.nextLinkId;
    context.nextLinkId += 1;
    return id;
}
/**
 * The input ports an instance declares, with the link each one receives.
 *
 * The `label` matters as much as the name: the definition calls its first input
 * `value`, while the instance labels it `prompt`. The label is what the author
 * meant a caller to see, so it is carried into the flattened graph and becomes the
 * public parameter's name.
 *
 * @param node - the instance node.
 * @returns the ports, in declaration order.
 */
function inputPorts(node) {
    const ports = [];
    for (const entry of readArray(node, 'inputs')) {
        if (!isRecordLike(entry))
            continue;
        const name = readString(entry, 'name') ?? '';
        const label = readString(entry, 'label') ?? name;
        const link = entry['link'];
        ports.push({ name, label, link: typeof link === 'number' ? link : undefined });
    }
    return ports;
}
/**
 * The value an instance binds to each of its ports.
 *
 * `widgets_values_named` wins when present: it is keyed by port name, so it cannot
 * be thrown off by an author reordering ports. The positional array is the
 * fallback for older revisions, read in the declaration order of the definition's
 * `inputs[]`.
 *
 * @param node - the instance node.
 * @param definition - the subgraph it instantiates.
 * @param ports - its input ports.
 * @param links - the links available at this level, for resolving connected ports.
 * @returns a binding per port name.
 */
function portBindings(node, definition, ports, links) {
    const named = isRecordLike(node['widgets_values_named']) ? node['widgets_values_named'] : undefined;
    const positional = Array.isArray(node['widgets_values']) ? node['widgets_values'] : [];
    const bindings = new Map();
    ports.forEach((port, index) => {
        if (port.link !== undefined) {
            const link = links.find((candidate) => candidate.id === port.link);
            if (link !== undefined) {
                bindings.set(port.name, { source: { originId: link.originId, originSlot: link.originSlot }, bound: true });
                return;
            }
        }
        if (named !== undefined && Object.hasOwn(named, port.name)) {
            bindings.set(port.name, { value: named[port.name], bound: true });
            return;
        }
        // The definition's own declaration order is the mapping the editor used, so
        // the positional array is read against that rather than against the ports.
        const declaredIndex = definition.inputNames.indexOf(port.name);
        const at = declaredIndex === -1 ? index : declaredIndex;
        if (at >= 0 && at < positional.length) {
            bindings.set(port.name, { value: positional[at], bound: true });
            return;
        }
        bindings.set(port.name, { bound: false });
    });
    return bindings;
}
/**
 * Expand one subgraph instance into the nodes it contains.
 *
 * @param node - the instance node.
 * @param definition - the subgraph it instantiates.
 * @param prefix - the id prefix for this instance, already including ancestors.
 * @param incoming - the links available where the instance sits.
 * @param context - shared expansion state.
 * @returns the expanded nodes and where its outputs come from.
 */
function expandInstance(node, definition, prefix, incoming, context) {
    const ports = inputPorts(node);
    const bindings = portBindings(node, definition, ports, incoming);
    /** The author's label for each public input, by the definition's own port name. */
    const labels = new Map(ports.map((port) => [port.name, port.label]));
    const nodes = [];
    /** For each nested instance placed here, where its outputs really come from. */
    const nestedOutputs = new Map();
    // Port names in proxy-slot order, which is how an input-proxy link names its port.
    const nameForSlot = (slot) => definition.inputNames[slot];
    for (const internal of definition.nodes) {
        const internalId = internal['id'];
        if (typeof internalId !== 'string' && typeof internalId !== 'number')
            continue;
        const id = String(internalId);
        // The proxies are wiring, not nodes: their edges have been translated above.
        if (id === definition.inputNodeId || id === definition.outputNodeId)
            continue;
        const classType = readString(internal, 'type');
        if (classType === undefined)
            continue;
        // A definition may itself place instances of another subgraph. Recursing is
        // not enough on its own: this instance's ports may be fed by the *enclosing*
        // subgraph's public inputs, which have to be resolved one level up before the
        // inner definition can be expanded, or the inner expansion wires itself to a
        // proxy node that will not exist in the flattened graph.
        const nested = context.definitions.get(classType);
        if (nested !== undefined) {
            const translated = translateNestedInstance(internal, definition, bindings, nameForSlot, context, prefix);
            const nestedExpansion = expandInstance(translated, nested, `${prefix}${id}:`, [...definition.links, ...translated.extraLinks], context);
            nodes.push(...nestedExpansion.nodes);
            nestedOutputs.set(id, nestedExpansion.outputs);
            continue;
        }
        const clone = { ...internal, id: `${prefix}${id}` };
        const rewritten = [];
        const literals = {};
        for (const port of inputPorts(internal)) {
            const link = port.link === undefined ? undefined : definition.links.find((entry) => entry.id === port.link);
            if (link === undefined) {
                // An unconnected port keeps whatever the definition had; the editor
                // stores that in `widgets_values`, which the converter reads.
                rewritten.push({ name: port.name, link: undefined });
                continue;
            }
            if (link.originId === definition.inputNodeId) {
                // This port is one of the subgraph's public inputs: take the instance's
                // binding — an outside link, or the per-instance value.
                const portName = nameForSlot(link.originSlot) ?? port.name;
                const binding = bindings.get(portName) ?? { bound: false };
                if (binding.source !== undefined) {
                    const id2 = allocateLink(context);
                    rewritten.push({ name: port.name, link: id2 });
                    context.synthesized.push([
                        id2,
                        binding.source.originId,
                        binding.source.originSlot,
                        `${prefix}${id}`,
                        rewritten.length - 1,
                        link.type,
                    ]);
                }
                else if (binding.bound) {
                    // A promoted widget: the value belongs on the node input itself, not on
                    // a link, which is exactly what "override" means here.
                    literals[port.name] = binding.value;
                    // Carry the instance's own label onto the target so the author's name for
                    // this input survives flattening. Without it the analysis sees an
                    // unnamed literal and has to withhold the input, which is how a workflow
                    // whose prompt is a promoted `value` port ends up advertising no
                    // capability at all.
                    const label = labels.get(portName) ?? portName;
                    if (clone['title'] === undefined && label.length > 0)
                        clone['title'] = label;
                }
                continue;
            }
            const id2 = allocateLink(context);
            rewritten.push({ name: port.name, link: id2 });
            context.synthesized.push([
                id2,
                `${prefix}${link.originId}`,
                link.originSlot,
                `${prefix}${id}`,
                rewritten.length - 1,
                link.type,
            ]);
        }
        if (Object.keys(literals).length > 0) {
            // Promote the override into the node's widget values so the existing
            // converter assigns it by name rather than by position.
            const named = isRecordLike(clone['widgets_values_named']) ? { ...clone['widgets_values_named'] } : {};
            clone['widgets_values_named'] = { ...named, ...literals };
        }
        clone['inputs'] = rewritten;
        nodes.push(clone);
    }
    // Where an internal value lands when it reaches the output proxy. Resolved
    // *after* the nodes are expanded, because an output may be fed by a nested
    // instance — in which case the real source is inside that instance, and
    // prefixing the instance id would name a node the flattened graph no longer has.
    const outputs = new Map();
    for (const link of definition.links) {
        if (link.targetId !== definition.outputNodeId)
            continue;
        const nested = nestedOutputs.get(link.originId);
        if (nested !== undefined) {
            const real = nested.get(link.originSlot);
            if (real !== undefined)
                outputs.set(link.targetSlot, { originId: real.originId, originSlot: real.originSlot });
            continue;
        }
        outputs.set(link.targetSlot, { originId: `${prefix}${link.originId}`, originSlot: link.originSlot });
    }
    context.expanded += 1;
    return { nodes, outputs };
}
/**
 * Resolve a nested instance's ports against the subgraph that encloses it.
 *
 * A definition that places an instance of another subgraph usually feeds it from
 * its *own* public inputs. Those ports cannot be resolved inside the inner
 * definition — the proxy they point at belongs to the outer one — so each is
 * translated one level up first: a port fed by an outer input becomes either the
 * value bound to that input (injected as the nested instance's own override) or a
 * link from whatever the outer input was connected to.
 *
 * @param node - the nested instance node.
 * @param definition - the enclosing subgraph.
 * @param bindings - the enclosing instance's bindings, by port name.
 * @param nameForSlot - the enclosing definition's port name for a proxy slot.
 * @param context - shared expansion state.
 * @param prefix - the id prefix the nested instance will be expanded under.
 * @returns a copy of the node with resolved ports, plus any links that were invented.
 */
function translateNestedInstance(node, definition, bindings, nameForSlot, context, prefix) {
    const named = isRecordLike(node['widgets_values_named']) ? { ...node['widgets_values_named'] } : {};
    const extraLinks = [];
    const ports = inputPorts(node).map((port) => {
        if (port.link === undefined)
            return { ...port };
        const link = definition.links.find((entry) => entry.id === port.link);
        if (link === undefined || link.originId !== definition.inputNodeId)
            return { ...port };
        const outerPort = nameForSlot(link.originSlot) ?? port.name;
        const binding = bindings.get(outerPort);
        if (binding?.source !== undefined) {
            const id = allocateLink(context);
            extraLinks.push({
                id,
                originId: binding.source.originId,
                originSlot: binding.source.originSlot,
                targetId: String(node['id']),
                targetSlot: 0,
                type: link.type,
            });
            return { ...port, link: id };
        }
        if (binding !== undefined && binding.bound)
            named[port.name] = binding.value;
        return { ...port, link: undefined };
    });
    return {
        ...node,
        inputs: ports.map((port) => ({ name: port.name, label: port.label, link: port.link ?? null })),
        widgets_values_named: named,
        extraLinks,
    };
}
/**
 * Flatten every subgraph instance in a UI export.
 *
 * The result is a document the existing UI→API converter understands: real node
 * classes, ids unique across the whole graph, and links in the positional array
 * form. A document with no subgraphs is returned with its nodes untouched, so
 * existing workflows behave exactly as before.
 *
 * @param document - the UI-format document.
 * @returns the flattened nodes, links and notes.
 */
export function flattenUiSubgraphs(document) {
    const definitions = readSubgraphDefinitions(document);
    const rawNodes = readArray(document, 'nodes').filter(isRecordLike);
    const rawLinks = readArray(document, 'links')
        .map((entry) => readUiLink(entry))
        .filter((link) => link !== undefined);
    if (definitions.size === 0) {
        return {
            nodes: rawNodes,
            links: rawLinks.map((link) => [link.id, link.originId, link.originSlot, link.targetId, link.targetSlot, link.type]),
            notes: [],
            expanded: false,
        };
    }
    const highestLinkId = rawLinks.reduce((highest, link) => Math.max(highest, link.id), 0);
    const context = {
        definitions,
        notes: [],
        nextLinkId: highestLinkId + 1,
        expanded: 0,
        synthesized: [],
    };
    const nodes = [];
    const links = [];
    /** instance node id → its slot → the real internal source. */
    const instanceOutputs = new Map();
    for (const node of rawNodes) {
        const id = node['id'];
        const classType = readString(node, 'type');
        const definition = classType === undefined ? undefined : definitions.get(classType);
        if (definition === undefined || id === undefined) {
            // A plain node keeps its id; only the links around it are rewritten.
            nodes.push({ ...node, id: String(id) });
            continue;
        }
        const expansion = expandInstance(node, definition, `${String(id)}:`, rawLinks, context);
        instanceOutputs.set(String(id), expansion.outputs);
        nodes.push(...expansion.nodes);
    }
    const instanceIds = new Set(instanceOutputs.keys());
    for (const link of rawLinks) {
        // A link into an instance was consumed by the expansion; one out of an
        // instance is re-pointed at the internal node that actually produces it.
        if (instanceIds.has(link.targetId))
            continue;
        const source = instanceIds.has(link.originId)
            ? instanceOutputs.get(link.originId)?.get(link.originSlot)
            : { originId: link.originId, originSlot: link.originSlot };
        if (source === undefined)
            continue;
        links.push([link.id, source.originId, source.originSlot, link.targetId, link.targetSlot, link.type]);
    }
    for (const entry of context.synthesized) {
        if (Array.isArray(entry)) {
            links.push(entry);
        }
    }
    return {
        nodes,
        links,
        notes: context.expanded === 0
            ? []
            : [
                `${context.expanded} subgraph instance(s) were expanded into their contents: ` +
                    `${[...definitions.values()].map((entry) => entry.name).join(', ')}.`,
            ],
        expanded: context.expanded > 0,
    };
}
/**
 * Whether a document contains subgraph definitions at all.
 * @param document - the UI-format document.
 * @returns true when there is at least one definition.
 */
export function hasSubgraphs(document) {
    const container = document['definitions'];
    return isRecordLike(container) && readArray(container, 'subgraphs').length > 0;
}
