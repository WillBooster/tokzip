import { javaModule } from '../generated/java.ts';
import { registerLanguageModule } from '../moduleRegistry.ts';

// Self-registers on import; the model is validated at registration.
registerLanguageModule(javaModule);

export { javaModule };
