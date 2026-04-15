package com.projet.ecgfront.service;

import java.io.IOException;
import java.net.URI;
import java.time.Duration;
import java.util.Map;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.RequestEntity;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

@Service
public class AiGatewayService {

    private final RestTemplate restTemplate;
    private final String aiServiceUrl;
    private final String aiServiceUrlAll;

    public AiGatewayService(
            RestTemplateBuilder restTemplateBuilder,
            @Value("${ai.service.url}") String aiServiceUrl,
            @Value("${ai.service.url.all}") String aiServiceUrlAll
    ) {
        this.restTemplate = restTemplateBuilder
                .setConnectTimeout(Duration.ofSeconds(10))
                .setReadTimeout(Duration.ofSeconds(60))
                .build();
        this.aiServiceUrl = aiServiceUrl;
        this.aiServiceUrlAll = aiServiceUrlAll;
    }

    public ResponseEntity<Map<String, Object>> classify(String selectedModel, MultipartFile signalFile) throws IOException {
        return sendMultipart(aiServiceUrl, selectedModel, signalFile);
    }

    public ResponseEntity<Map<String, Object>> classifyAll(MultipartFile signalFile) throws IOException {
        return sendMultipart(aiServiceUrlAll, null, signalFile);
    }

    private ResponseEntity<Map<String, Object>> sendMultipart(String targetUrl, String model, MultipartFile signalFile) throws IOException {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        ByteArrayResource resource = new ByteArrayResource(signalFile.getBytes()) {
            @Override
            public String getFilename() {
                String original = signalFile.getOriginalFilename();
                return (original == null || original.isBlank()) ? "signal.txt" : original;
            }
        };

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        if (model != null) {
            body.add("model", model);
        }
        body.add("signal", resource);

        RequestEntity<MultiValueMap<String, Object>> request = RequestEntity
            .post(URI.create(targetUrl))
            .headers(headers)
            .body(body);

        return restTemplate.exchange(request, new ParameterizedTypeReference<>() {});
    }
}
