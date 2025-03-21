const http = require('http');
const fs = require('fs');
const path = require('path');
const neo4j = require('neo4j-driver');

// Neo4j connection details
const uri = "bolt://localhost:7687";
const user = "neo4j";
const password = "12345678"; // Updated to match your actual password

// Create Neo4j driver
const driver = neo4j.driver(uri, neo4j.auth.basic(user, password));

// Create basic HTTP server
const server = http.createServer(async (req, res) => {
    if (req.url === '/') {
        // Serve the main HTML page
        fs.readFile(path.join(__dirname, 'index.html'), (err, content) => {
            if (err) {
                res.writeHead(500);
                res.end('Error loading index.html');
                return;
            }
            res.writeHead(200, { 'Content-Type': 'text/html' });
            res.end(content);
        });
    } else if (req.url === '/data') {
        try {
            // Connect to Neo4j and fetch graph data
            const session = driver.session();
            const result = await session.run(`
                MATCH (n:Article)
                OPTIONAL MATCH (n)-[r]->(m:Article)
                RETURN n, r, m
            `);

            // Transform Neo4j result into visualization format
            const nodes = new Set();
            const links = [];

            result.records.forEach(record => {
                const sourceNode = record.get('n');
                const targetNode = record.get('m');
                const relationship = record.get('r');

                if (sourceNode) {
                    nodes.add({
                        id: sourceNode.identity.low,
                        label: sourceNode.properties.title,
                        content: sourceNode.properties.content,
                        summary: sourceNode.properties.summary
                    });
                }

                if (targetNode) {
                    nodes.add({
                        id: targetNode.identity.low,
                        label: targetNode.properties.title,
                        content: targetNode.properties.content,
                        summary: targetNode.properties.summary
                    });
                }

                if (relationship) {
                    links.push({
                        source: sourceNode.identity.low,
                        target: targetNode.identity.low,
                        type: relationship.type,
                        confidence: relationship.properties.confidence
                    });
                }
            });

            await session.close();

            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({
                nodes: Array.from(nodes),
                links: links
            }));
        } catch (error) {
            console.error('Error fetching data from Neo4j:', error);
            res.writeHead(500);
            res.end(JSON.stringify({ error: 'Error fetching data from Neo4j' }));
        }
    } else {
        res.writeHead(404);
        res.end('Not found');
    }
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
}); 