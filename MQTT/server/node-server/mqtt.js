var config;
var server;
const mosca = require('mosca');

var LANDMARKS = [
    {
        timestamp: 1.2,
        landmark: "A zoo nearby"
    },
    {
        timestamp: 2.4,
        landmark: "You are on the motorway looking at a park"
    },
    {
        timestamp: 2.7,
        landmark: "The section of land consists of tourists spots"
    }
];

module.exports = {
    configure: function (c) {
        config = c;
    },

    start: function () {
        server = new mosca.Server({
            port: config.mqtt.port,
            http: config.mqtt.http,
        });

        server.on('ready', setup);
        server.on('clientConnected', connected);
        server.on('clientDisconnected', disconnected);
        server.on('published', published);
        server.on('subscribed', subscribed);
        server.on('unsubscribed', unsubscribed);
    },

    publish: function (topic, message) {
        var payload = {
            topic: topic,
            payload: message,
            qos: 0,
            retain: false
        };

        server.publish(payload, function () {
            console.log('Published callback complete.');
        });
    }
};

function setup() {
    console.log('Mosca server started.');
}

function connected(client) {
    console.log(`Client ${client.id} connected`);
}

function subscribed(topic, client) {
    console.log(`Client ${client.id} subscribed to ${topic}.`);
}

function unsubscribed(topic, client) {
    console.log(`Client ${client.id} unsubscribed from ${topic}.`);
}

function disconnected(client) {
    console.log(`Client ${client.id}`);
}

function published(packet, client) {

    landmarks = LANDMARKS.filter((landmark) => {
        if (packet.payload.timestamp <= landmark.timestamp) {
           return true; 
        }
        return false;
    });

    console.log("The dashcam said: " + landmarks[0].landmark + ", the dashcam is exiting now.");

}
